"""
mtat_features.py -- Task 3 preprocessing, start to finish, in one file.

Same shape as audio_features.py (the GTZAN/Task 2 version) -- same constant
names, same function names, same output columns -- with the three changes MTAT
forces:

    1. MULTI-LABEL targets instead of one genre int. The 188-tag vocabulary is
       partitioned into three disjoint groups: instrument/vocal tags become the
       BERT input text, genre + mood tags become the prediction targets.
    2. GROUPED splits. MTAT's 25,863 clips come from only 5,405 source songs
       (~5.8 clips per song), so the standard folder split puts different
       29-second slices of the SAME recording in train and test.
    3. No raw waveforms. 17,623 usable clips x 5 segments is ~88k rows;
       storing waveforms too would produce a ~28 GB parquet.

Pipeline:
    1. Pull annotations_final.csv, clip_info_final.csv and mp3.zip from the
       confit/magnatagatune HF repo; unzip the audio.
    2. Partition the 188 tags into instrument / genre / mood groups, drop tags
       below MIN_POSITIVES (targets only -- instrument tags keep everything),
       and report anything unassigned for review.
    3. Load each clip mono at 22,050 Hz, peak-normalize, cut into 5 s
       non-overlapping segments (29.1 s clip -> 5 segments).
    4. Per segment: chroma_stft (12 x 216) and MFCC (13 x 216), pooled to
       chroma_pooled (24-d) and mfcc_pooled (26-d).
    5. Build the BERT input sentence from each clip's instrument/vocal tags.
    6. Song-grouped 70/15/15 split (GROUP_BY selects song / artist / clip).
    7. Write every stage to parquet, same as the GTZAN pipeline:
           segmented_audio_data.parquet         frame-level features
           segmented_audio_data_pooled.parquet  + pooled features
           train.parquet / val.parquet / test.parquet
           label_space.json

Every stage is resumable: if a stage's parquet already exists it is reused
rather than recomputed, so a crash three hours into feature extraction does not
cost you the whole run. Delete the file to force a rebuild.

Every stage is a datasets.map(batched=True, batch_size=BATCH_SIZE) -- the same
pattern as your GTZAN notebooks, including the one-clip-in / many-segments-out
flattening map. Arrow memory-maps to disk, so peak RAM stays at one batch
regardless of corpus size. ~88,000 segments x (12+13) x 216 frames is roughly
1.9 GB of frame features; holding that in one pandas DataFrame would need well
over 16 GB.

Note on resampling: MTAT audio is natively 16 kHz. It is loaded at 22,050 Hz so
that a 5 s segment gives exactly 216 frames, matching the GTZAN pipeline. See
SAMPLE_RATE and MEL_FMAX in the configuration block.

There are no command-line arguments. Set the constants in the CONFIGURATION
block below, then:

    python src/mtat_features.py

or, from a notebook:

    from mtat_features import main
    train_df, val_df, test_df = main()

Run inspect_tags() on its own first -- it does step 2 only, in seconds, and
tells you whether the mood group survives MIN_POSITIVES before you commit an
hour to feature extraction.

Standardization is NOT done here, for the same reason as before: the scaler is
fit on train only, inside mtat_graphs.load_splits().
"""

from __future__ import annotations

import json
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value, load_dataset


# ===========================================================================
# 0. CONFIGURATION -- this is the only block you need to edit
# ===========================================================================

# --- source -----------------------------------------------------------------
HF_REPO = "confit/magnatagatune"
AUDIO_DIR = Path("data/raw/mtat/mp3")      # mp3.zip is extracted here
OUT_DIR = Path("data/processed/mtat")

# Each stage of the dataset is saved, exactly as in the GTZAN pipeline.
SEGMENTED_PARQUET = OUT_DIR / "segmented_audio_data.parquet"
POOLED_PARQUET = OUT_DIR / "segmented_audio_data_pooled.parquet"

# --- batching ---------------------------------------------------------------
# Everything runs through datasets.map(batched=True), the same pattern as your
# GTZAN notebooks. Arrow memory-maps its output to disk, so peak RAM is set by
# BATCH_SIZE x WRITER_BATCH_SIZE, not by the size of the dataset -- which is
# what makes the ~2 GB of frame features runnable on a laptop.
BATCH_SIZE = 32           # clips (stage 1) or segments (stage 2) per map batch
WRITER_BATCH_SIZE = 200   # rows Arrow buffers before flushing to disk
NUM_PROC = 1              # raise to parallelize; librosa + fork can be flaky on macOS,
                          # so try 4 and drop back to 1 if workers hang
USE_EXPLICIT_FEATURES = True   # set False to fall back to schema inference

# Cap the number of clips for a fast development pass. None = all 17,623 that
# survive the target-tag filter. Start with 2000 to check the pipeline.
MAX_CLIPS = None

# --- audio ------------------------------------------------------------------
# MTAT mp3s are natively 16 kHz mono 32 kbps, so resampling to 22,050 adds no
# new information -- everything above the original 8 kHz Nyquist is
# interpolation. It is still the right choice here, for one concrete reason:
# at 22,050 Hz a 5 s segment is 110,250 samples, which at hop 512 gives exactly
# 216 frames -- the same as the GTZAN segments in Task 2. Identical frame
# counts mean the pooled statistics are averaged over the same number of frames
# in both tasks, so the Task 2 and Task 3 feature vectors stay directly
# comparable. At 16 kHz they would have been 157 frames and quietly different.
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 5                        # 29.1 s clip -> 5 segments
PEAK_NORMALIZE = True

# Upper bound on the MFCC mel filterbank. Left at sr/2 = 11,025 Hz, the top mel
# bands would sit entirely above the source material's 8 kHz ceiling and
# describe resampling artifacts rather than music -- which would ALSO break the
# comparability with GTZAN that the resampling was for. Capping at the original
# Nyquist keeps every filter over real content. Set to None to use sr/2.
# chroma_stft takes no fmax, but it is pitch-class folded and there is
# effectively no energy up there, so it is unaffected.
MEL_FMAX = 8000

# --- features ---------------------------------------------------------------
# Identical to the GTZAN pipeline. At 22,050 Hz a 5 s segment is 110,250
# samples, so hop 512 gives 1 + 110250//512 = 216 frames per segment -- the
# same shape the GTZAN features have.
N_CHROMA = 12                              # -> chroma_pooled is 24-d
N_MFCC = 13                                # -> mfcc_pooled is 26-d
N_FFT = 2048
HOP_LENGTH = 512
STD_DDOF = 1                               # matches torch.std, as before

# --- what to store ----------------------------------------------------------
# KEEP_FRAME_FEATURES stores the full (12, 216) and (13, 216) matrices, so the
# saved dataset matches the GTZAN one column for column. That is about 1.9 GB
# across ~88k segments -- large but the same order as your existing
# segmented_audio_data.parquet, and it is what lets you revisit pooling or do
# frame-level chord estimation later without re-extracting anything.
#
# KEEP_WAVEFORM stays off. Raw samples would add roughly 26 GB, and nothing
# downstream of Task 3 reads them -- only the mel-spectrogram CNN baseline would,
# and that baseline already exists on GTZAN.
KEEP_FRAME_FEATURES = True
KEEP_WAVEFORM = False

# --- label space ------------------------------------------------------------
# Floor on TARGET tags only (genre + mood). A rare target tag is genuinely
# harmful: split 70/15/15 it leaves a handful of test positives, its per-tag F1
# becomes noise, and Macro-F1 moves for reasons unrelated to the model.
MIN_POSITIVES = 70

# Instrument tags are the BERT *input*, never predicted, so a rare one costs
# nothing -- it just adds a word to one clip's sentence and gives the text
# encoder more to work with. Kept at 0 deliberately; raise it only if the
# generated sentences get unwieldy.
MIN_POSITIVES_INSTRUMENT = 0

# --- splits -----------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15                            # test gets the remainder
# What the split groups on. "song" is the default and the minimum defensible
# choice; see split_by_group() for the full argument.
#   "song"   -- no source recording spans two splits (5,405 groups)
#   "artist" -- stricter, no artist spans two splits (230 groups)
#   "clip"   -- no grouping; reproduces the standard folder-split protocol
GROUP_BY = "song"
SEED = 42

VERBOSE = True


# ===========================================================================
# 1. Tag partition
# ===========================================================================
#
# The three groups must stay DISJOINT. That is the whole point: if the text fed
# to BERT overlaps the prediction targets, BERT-only saturates, the GNN branch
# contributes nothing measurable, and the Task 3 ablation stops measuring
# anything. Instrumentation is informative about genre without being a
# giveaway, which is the regime where cross-attention can actually beat both
# single modalities.

INSTRUMENT_TAGS = [
    # strings / plucked
    "guitar", "guitars", "electric guitar", "classical guitar", "acoustic",
    "acoustic guitar", "no strings",
    "banjo", "sitar", "harp", "lute", "fiddle", "plucking",
    "violin", "violins", "cello", "strings", "string", "no violin", "no guitar",
    # keys
    "piano", "no piano", "piano solo", "harpsichord", "harpsicord", "organ",
    "keyboard", "synth", "synthesizer", "electric",
    # winds / brass
    "flute", "flutes", "no flute", "oboe", "clarinet", "sax", "trumpet",
    "horn", "horns", "wind",
    # percussion / low end
    "drums", "drum", "no drums", "percussion", "bongos", "clapping",
    "beat", "beats", "no beat", "bass", "bells", "chimes",
    # ensemble / texture
    "orchestra", "orchestral", "instrumental", "solo",
    # non-instrument sound sources. These describe what is audible in the
    # recording, which is the same kind of information as instrumentation, and
    # they carry no genre signal -- so they are safe in the BERT input.
    "water", "birds",
    # voice
    "vocal", "vocals", "no vocals", "no vocal", "voice", "no voice", "voices",
    "singer", "singing", "no singing", "no singer", "male singer",
    "female singer", "man singing", "woman singing", "female singing",
    "male vocal", "female vocal",
    "male voice", "female voice", "male vocals", "female vocals",
    "male", "female", "man", "woman", "men", "women", "girl",
    "choir", "choral", "chorus", "chant", "chanting", "monks", "duet",
    "operatic", "male opera", "female opera", "talking",
]

GENRE_TAGS = [
    "classical", "baroque", "opera", "medieval",
    "rock", "hard rock", "soft rock", "heavy metal", "metal", "punk",
    "pop", "dance", "disco", "house", "trance", "techno",
    "electronic", "electro", "industrial",
    "jazz", "blues", "funky",
    "country", "folk", "reggae", "celtic", "irish", "spanish",
    "ambient", "new age",
    "rap", "hip hop",
    "indian", "india", "eastern", "middle eastern", "oriental", "arabic",
    "tribal", "world", "foreign",
]

MOOD_TAGS = [
    # dynamics
    "loud", "quiet", "soft", "hard", "heavy", "light", "deep",
    # tempo / motion
    "slow", "fast", "fast beat", "upbeat", "repetitive",
    # affect / character
    "dark", "calm", "mellow", "sad", "eerie", "scary", "airy", "spacey",
    "space", "echo", "weird", "strange", "drone",
    "silence",
]

# Excluded on purpose. "not classical" / "not rock" / "not opera" are direct
# negations of GENRE_TAGS entries, so putting them in the BERT input would leak
# the target straight into the text. "english" / "not english" is a language
# cue, unrelated to either group.
EXCLUDED_TAGS = [
    # Direct negations of GENRE_TAGS entries -- putting these in the BERT input
    # would pipe the target straight into the text.
    "not classical", "not rock", "not opera",
    # Language cues, unrelated to either target group.
    "english", "not english",
    # Era descriptors. "modern" (327) and "old" (55) are frequent but mean
    # nothing consistent coming from a game player, and neither is a genre or
    # a mood.
    "modern", "old",
    # Annotation noise. "lol" (54) is a tagging-game artifact and a useful
    # concrete example for the label-quality paragraph in the report.
    "lol",
    # Ambiguous. "jungle" (53) is either the breakbeat genre or literal jungle
    # sounds, and on a Magnatune catalogue -- indie, classical and world, with
    # "tribal", "birds" and "water" also in the vocabulary -- the ambient sense
    # is at least as likely. Unresolvable from the label alone, so it stays out
    # rather than polluting the genre targets.
    "jungle",
]


def partition_tags(ann: pd.DataFrame, min_positives: int = MIN_POSITIVES,
                   min_positives_instrument: int = MIN_POSITIVES_INSTRUMENT,
                   verbose: bool = VERBOSE):
    """
    Intersects the three keyword lists above with the tag columns that actually
    exist in annotations_final.csv, applies the frequency floors, and reports
    anything left unassigned so you can decide where it belongs.

    The floors differ by group on purpose. Genre and mood are predicted, so a
    rare tag there poisons Macro-F1. Instrument tags only ever appear in the
    BERT input sentence, so a rare one is free information -- there is no
    reason to throw it away.

    Returns (instrument, genre, mood) as lists of surviving column names.
    """
    meta_cols = {"clip_id", "mp3_path"}
    tag_cols = [c for c in ann.columns if c not in meta_cols]
    counts = ann[tag_cols].sum()

    def keep(names, floor):
        return [t for t in names if t in tag_cols and counts[t] >= floor]

    instrument = keep(INSTRUMENT_TAGS, min_positives_instrument)
    genre = keep(GENRE_TAGS, min_positives)
    mood = keep(MOOD_TAGS, min_positives)

    assigned = set(INSTRUMENT_TAGS) | set(GENRE_TAGS) | set(MOOD_TAGS) | set(EXCLUDED_TAGS)
    unassigned = sorted((t for t in tag_cols if t not in assigned),
                        key=lambda t: -counts[t])
    dropped = sorted(
        ((t, int(counts[t])) for t in GENRE_TAGS + MOOD_TAGS
         if t in tag_cols and counts[t] < min_positives),
        key=lambda kv: -kv[1],
    )

    overlap = (set(instrument) & set(genre)) | (set(instrument) & set(mood)) | (set(genre) & set(mood))
    assert not overlap, f"tag groups must be disjoint, found: {sorted(overlap)}"

    if verbose:
        print(f"{len(tag_cols)} tag columns | target floor >= {min_positives}, "
              f"instrument floor >= {min_positives_instrument}\n")
        for name, group in [("instrument (-> BERT text, unfiltered)", instrument),
                            ("genre (-> target)", genre),
                            ("mood (-> target)", mood)]:
            print(f"{name}: {len(group)} tags")
            print("   " + ", ".join(f"{t}({int(counts[t])})" for t in
                                    sorted(group, key=lambda t: -counts[t])))
            print()
        if dropped:
            print(f"TARGET tags dropped by the floor ({len(dropped)}) -- this is "
                  f"what MIN_POSITIVES={min_positives} costs you:")
            print("   " + ", ".join(f"{t}({c})" for t, c in dropped))
            print()
        if unassigned:
            print(f"UNASSIGNED ({len(unassigned)}) -- decide where these belong, "
                  f"or add them to EXCLUDED_TAGS:")
            print("   " + ", ".join(f"{t}({int(counts[t])})" for t in unassigned))
            print()
        if not mood:
            print("WARNING: no mood tags survived. Either lower MIN_POSITIVES or "
                  "fall back to MTG-Jamendo.\n")

    return instrument, genre, mood


def inspect_tags(min_positives: int = MIN_POSITIVES):
    """
    Step 2 only -- downloads the 21.5 MB annotations file and prints the
    partition. Seconds, not an hour. Run this before committing to a full pass.
    """
    ann, _ = load_annotations()
    return partition_tags(ann, min_positives)


# ===========================================================================
# 2. Download + load metadata
# ===========================================================================

def load_annotations(repo: str = HF_REPO, verbose: bool = VERBOSE):
    """
    Returns (annotations, clip_info) as DataFrames.

    Both files are TAB-separated despite the .csv extension.
    annotations_final.csv: clip_id + 188 binary tag columns + mp3_path.
    clip_info_final.csv:   clip_id, title, artist, album, mp3_path, ...
                           the artist column is what makes grouped splits work.
    """
    from huggingface_hub import hf_hub_download

    ann_path = hf_hub_download(repo, "annotations_final.csv", repo_type="dataset")
    info_path = hf_hub_download(repo, "clip_info_final.csv", repo_type="dataset")

    ann = pd.read_csv(ann_path, sep="\t")
    info = pd.read_csv(info_path, sep="\t")

    if verbose:
        print(f"annotations: {ann.shape}, clip_info: {info.shape}")
        print(f"clip_info columns: {info.columns.tolist()}\n")
    return ann, info


def ensure_audio(repo: str = HF_REPO, audio_dir: Path = AUDIO_DIR,
                 verbose: bool = VERBOSE) -> Path:
    """Downloads mp3.zip (2.97 GB) once and extracts it. Skips if already there."""
    from huggingface_hub import hf_hub_download

    audio_dir = Path(audio_dir)
    if audio_dir.exists() and any(audio_dir.rglob("*.mp3")):
        if verbose:
            print(f"audio already extracted at {audio_dir}")
        return audio_dir

    if verbose:
        print("downloading mp3.zip (2.97 GB) -- one time only")
    zip_path = hf_hub_download(repo, "mp3.zip", repo_type="dataset")
    audio_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(audio_dir)
    if verbose:
        print(f"extracted {len(list(audio_dir.rglob('*.mp3')))} mp3s to {audio_dir}")
    return audio_dir


# ===========================================================================
# 3. Segmentation  (identical to the GTZAN version)
# ===========================================================================

def segment_waveform(y: np.ndarray, sr: int = SAMPLE_RATE,
                     seconds: int = SEGMENT_SECONDS):
    """
    Consecutive non-overlapping windows; an incomplete final window is dropped
    rather than zero-padded, so every segment yields the same frame count and
    the pooled statistics are computed over an equal number of frames for every
    node. MTAT clips are 29.1 s, so 5 s windows give 5 segments and discard the
    trailing 4.1 s.
    """
    window = seconds * sr
    return [y[i:i + window] for i in range(0, len(y), window)
            if len(y[i:i + window]) == window]


# ===========================================================================
# 4. Feature extraction + pooling  (identical to the GTZAN version)
# ===========================================================================

def extract_chroma(segment: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """(12, 216) chroma_stft -- pitch-class energy, i.e. harmonic content."""
    return librosa.feature.chroma_stft(
        y=segment, sr=sr, n_chroma=N_CHROMA, n_fft=N_FFT, hop_length=HOP_LENGTH
    ).astype(np.float32)


def extract_mfcc(segment: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """(13, 216) MFCCs -- spectral envelope, i.e. timbre."""
    kwargs = {} if MEL_FMAX is None else {"fmax": MEL_FMAX}
    return librosa.feature.mfcc(
        y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
        **kwargs
    ).astype(np.float32)


def pool_mean_std(feature_array: np.ndarray) -> np.ndarray:
    """
    (n_bins, T) -> (2 * n_bins,) as [means..., stds...].

    The std half separates a sustained pad from a busy percussive passage that
    averages to the same place. ddof=1 matches torch.std, so these values stay
    comparable with the GTZAN pooled features.
    """
    arr = np.asarray(feature_array, dtype=np.float32)
    return np.concatenate([
        arr.mean(axis=1),
        arr.std(axis=1, ddof=STD_DDOF),
    ]).astype(np.float32)


def as_matrix(value) -> np.ndarray:
    """
    Frame features come back from parquet as a list of rows, not a clean 2-D
    array. Rebuild the (n_bins, T) matrix regardless of which form it is in.
    """
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.stack([np.asarray(r, dtype=np.float32) for r in arr])
    return arr.astype(np.float32, copy=False)


# ===========================================================================
# 5. BERT input text
# ===========================================================================

def build_text(row, instrument_tags) -> str:
    """
    Renders a clip's instrument/vocal tags as one sentence, e.g.
        "A music clip featuring harpsichord, strings and no vocals."

    This is the H_text that the cross-attention layer attends over. Be honest
    about it in the report: template text has low lexical diversity and
    underuses BERT's pretraining compared with a real caption. The trade is
    deliberate -- MTAT has no free-form text, and a real caption here would have
    been written from the same listening pass as the target tags, which is the
    circularity that makes an ablation meaningless.
    """
    present = [t for t in instrument_tags if row.get(t, 0) == 1]
    if not present:
        return "A music clip."
    if len(present) == 1:
        body = present[0]
    else:
        body = ", ".join(present[:-1]) + " and " + present[-1]
    return f"A music clip featuring {body}."


# ===========================================================================
# 6. Schema
# ===========================================================================

def build_features(with_frames: bool = None, with_waveform: bool = None) -> Features:
    """
    Declared rather than inferred, for two reasons. It documents the saved
    dataset in one place, and it pins the feature matrices to float32 -- under
    inference they can round-trip through Python floats and land in the parquet
    as float64, silently doubling a 2 GB dataset.

    Set USE_EXPLICIT_FEATURES = False to fall back to inference if a version
    mismatch ever makes this schema disagree with what map() produces.
    """
    with_frames = KEEP_FRAME_FEATURES if with_frames is None else with_frames
    with_waveform = KEEP_WAVEFORM if with_waveform is None else with_waveform

    schema = {
        "track_id": Value("int64"),          # = MTAT clip_id
        "segment_id": Value("int32"),
        "song_id": Value("string"),          # source recording, for the split
        "artist": Value("string"),
        "text": Value("string"),             # BERT input, from instrument tags
        "genre_labels": Sequence(Value("int8")),
        "mood_labels": Sequence(Value("int8")),
        "is_silent": Value("bool"),
    }
    if with_frames:
        schema["chroma_features"] = Sequence(Sequence(Value("float32")))   # (12, 216)
        schema["mfcc_features"] = Sequence(Sequence(Value("float32")))     # (13, 216)
    if with_waveform:
        schema["segment"] = Sequence(Value("float32"))
    return Features(schema)


def pooled_features(base: Features) -> Features:
    schema = dict(base)
    schema["chroma_pooled"] = Sequence(Value("float32"))   # 24-d
    schema["mfcc_pooled"] = Sequence(Value("float32"))     # 26-d
    return Features(schema)


# ===========================================================================
# 7. Stage 1 -- segmented_audio_data.parquet (frame-level features)
# ===========================================================================

# librosa emits "Trying to estimate tuning from empty frequency set" once per
# near-silent segment, from chroma_stft's tuning estimation. It is not an error
# -- the chroma vector comes back as zeros, which is the correct answer for
# silence -- but at ~88,000 segments the warnings drown the progress output.
# Silenced here and counted instead, via SILENT_RMS below.
warnings.filterwarnings(
    "ignore", message="Trying to estimate tuning from empty frequency set")

SILENT_RMS = 1e-4   # segments below this are logged; see drop_silent_segments()
DROP_SILENT = False  # True removes them outright


def segment_and_extract(batch):
    """
    Batched map function. One clip in, several segments out -- the same
    flattening pattern as create_segmented_dataset() in your GTZAN notebook,
    where the returned lists are longer than the input batch.

    Clips whose mp3 is missing, empty or unreadable simply contribute no rows.
    MTAT ships a handful of zero-byte files, and a silent zero-length track
    would otherwise become a graph with no edges and a genre label attached to
    nothing.

    Near-silent segments are kept by default but flagged in an `is_silent`
    column. They are real -- fade-outs and gaps -- and dropping them would make
    some clips shorter than others, which changes the graph topology for
    reasons unrelated to musical structure. Set DROP_SILENT = True to remove
    them if they turn out to hurt.
    """
    out = {k: [] for k in ("track_id", "segment_id", "song_id", "artist", "text",
                           "genre_labels", "mood_labels", "is_silent")}
    if KEEP_FRAME_FEATURES:
        out["chroma_features"], out["mfcc_features"] = [], []
    if KEEP_WAVEFORM:
        out["segment"] = []

    for i, mp3_path in enumerate(batch["mp3_path"]):
        path = Path(AUDIO_DIR) / str(mp3_path)
        try:
            if not path.exists() or path.stat().st_size == 0:
                continue
            y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            continue

        if PEAK_NORMALIZE and np.any(y):
            y = librosa.util.normalize(y)

        for segment_id, seg in enumerate(segment_waveform(y)):
            silent = bool(np.sqrt(np.mean(seg ** 2)) < SILENT_RMS)
            if silent and DROP_SILENT:
                continue
            out["is_silent"].append(silent)
            out["track_id"].append(int(batch["track_id"][i]))
            out["segment_id"].append(segment_id)
            out["song_id"].append(str(batch["song_id"][i]))
            out["artist"].append(str(batch["artist"][i]))
            out["text"].append(str(batch["text"][i]))
            out["genre_labels"].append(list(batch["genre_labels"][i]))
            out["mood_labels"].append(list(batch["mood_labels"][i]))
            if KEEP_FRAME_FEATURES:
                out["chroma_features"].append(extract_chroma(seg))
                out["mfcc_features"].append(extract_mfcc(seg))
            if KEEP_WAVEFORM:
                out["segment"].append(seg.astype(np.float32))
    return out


def song_id_map(info: pd.DataFrame, verbose: bool = VERBOSE) -> dict:
    """
    Maps clip_id -> an identifier for the SOURCE RECORDING.

    MTAT clips are 29 s excerpts and ~5.8 of them come from each song, so this
    is the unit the split has to respect. `original_url` identifies the source
    track exactly when present; artist + album + title is the fallback, which is
    slightly coarser (it merges any two tracks sharing all three) but never
    splits one recording across groups, which is the property that matters.
    """
    cols = info.columns
    if "original_url" in cols:
        ids = info["original_url"].astype(str)
        source = "original_url"
    else:
        parts = [info[c].astype(str) for c in ("artist", "album", "title") if c in cols]
        if not parts:
            raise KeyError(f"cannot derive song_id from clip_info columns: {list(cols)}")
        ids = parts[0]
        for extra in parts[1:]:
            ids = ids + "::" + extra
        source = " + ".join(c for c in ("artist", "album", "title") if c in cols)

    mapping = dict(zip(info["clip_id"], ids))
    n_songs = max(len(set(mapping.values())), 1)
    if verbose:
        print(f"song_id from {source}: {n_songs} songs across {len(mapping)} clips "
              f"({len(mapping) / n_songs:.1f} clips/song)")
    return mapping


def build_clip_table(ann, info, instrument, genre, mood,
                     max_clips: int | None = MAX_CLIPS,
                     verbose: bool = VERBOSE) -> Dataset:
    """
    One row per CLIP -- the input to the stage 1 map. Everything the map needs
    is resolved here (song_id, artist, text, label vectors) so the map function
    itself touches nothing but audio, which is what lets NUM_PROC > 1 work.
    """
    song_by_clip = song_id_map(info, verbose)
    artist_by_clip = info.set_index("clip_id")["artist"].to_dict()

    df = ann.copy()
    # a clip with no positive target tag teaches the model nothing
    df = df[(df[genre].sum(axis=1) > 0) | (df[mood].sum(axis=1) > 0)]
    if max_clips:
        df = df.sample(n=min(max_clips, len(df)), random_state=SEED)

    clips = pd.DataFrame({
        "track_id": df["clip_id"].astype("int64").to_numpy(),
        "mp3_path": df["mp3_path"].astype(str).to_numpy(),
        "song_id": [str(song_by_clip.get(int(c), f"clip_{c}")) for c in df["clip_id"]],
        "artist": [str(artist_by_clip.get(int(c), "UNKNOWN")) for c in df["clip_id"]],
        "text": [build_text(row, instrument) for _, row in df.iterrows()],
        "genre_labels": [[int(row[t]) for t in genre] for _, row in df.iterrows()],
        "mood_labels": [[int(row[t]) for t in mood] for _, row in df.iterrows()],
    })
    if verbose:
        print(f"stage 1 input: {len(clips)} clips "
              f"({len(genre)} genre + {len(mood)} mood targets)")
        print(f"  example text: {clips.text.iloc[0]!r}")
    return Dataset.from_pandas(clips, preserve_index=False)


def build_segmented_dataset(ann, info, instrument, genre, mood,
                            out_path: Path = SEGMENTED_PARQUET,
                            verbose: bool = VERBOSE) -> Path:
    """Stage 1. Skipped entirely if out_path already exists."""
    out_path = Path(out_path)
    if out_path.exists():
        if verbose:
            print(f"stage 1: reusing {out_path} -- delete it to rebuild")
        return out_path

    clip_ds = build_clip_table(ann, info, instrument, genre, mood, verbose=verbose)
    seg_ds = clip_ds.map(
        segment_and_extract,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC if NUM_PROC > 1 else None,
        writer_batch_size=WRITER_BATCH_SIZE,
        remove_columns=clip_ds.column_names,
        features=build_features() if USE_EXPLICIT_FEATURES else None,
        desc="stage 1: segment + chroma/mfcc",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg_ds.to_parquet(out_path)

    if verbose:
        n_silent = int(np.sum(seg_ds["is_silent"]))
        print(f"\nstage 1: {len(seg_ds)} segments from "
              f"{len(set(seg_ds['track_id']))} clips "
              f"({n_silent} near-silent, {n_silent / max(len(seg_ds), 1):.2%})")
        print(f"  {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  {seg_ds}")
    return out_path


# ===========================================================================
# 8. Stage 2 -- segmented_audio_data_pooled.parquet (+ pooled features)
# ===========================================================================

def add_pooled(batch):
    """
    Batched map, identical in shape to features_pooling.ipynb: mean and std over
    the time axis, 12 bins -> 24-d and 13 coeffs -> 26-d.
    """
    return {
        "chroma_pooled": [pool_mean_std(as_matrix(c)) for c in batch["chroma_features"]],
        "mfcc_pooled": [pool_mean_std(as_matrix(m)) for m in batch["mfcc_features"]],
    }


def build_pooled_dataset(in_path: Path = SEGMENTED_PARQUET,
                         out_path: Path = POOLED_PARQUET,
                         verbose: bool = VERBOSE) -> Path:
    """
    Stage 2, kept separate from extraction because pooling is the part you are
    most likely to revisit. Swapping mean+std for something else is then a
    two-minute rerun over the saved frame features, not another pass over
    17,623 mp3s.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    if out_path.exists():
        if verbose:
            print(f"stage 2: reusing {out_path} -- delete it to rebuild")
        return out_path
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} missing -- run stage 1 first")

    seg_ds = load_dataset("parquet", data_files=str(in_path))["train"]
    pooled_ds = seg_ds.map(
        add_pooled,
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_PROC if NUM_PROC > 1 else None,
        writer_batch_size=WRITER_BATCH_SIZE,
        features=pooled_features(seg_ds.features) if USE_EXPLICIT_FEATURES else None,
        desc="stage 2: pooling",
    )
    pooled_ds.to_parquet(out_path)

    if verbose:
        row = pooled_ds[0]
        print(f"\nstage 2: {len(pooled_ds)} rows -> {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  chroma_features {as_matrix(row['chroma_features']).shape}, "
              f"mfcc_features {as_matrix(row['mfcc_features']).shape}")
        print(f"  chroma_pooled {len(row['chroma_pooled'])}-d, "
              f"mfcc_pooled {len(row['mfcc_pooled'])}-d")
    return out_path


# ===========================================================================
# 9. Stage 3 -- grouped split -> train/val/test.parquet
# ===========================================================================

def load_pooled(in_path: Path = POOLED_PARQUET):
    """Opens the stage 2 parquet. Arrow memory-maps it, so this is not a read."""
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} missing -- run stages 1 and 2 first")
    return load_dataset("parquet", data_files=str(in_path))["train"]


def attach_labels(ds, ann, instrument, genre, mood, verbose: bool = VERBOSE):
    """
    Rewrites `text`, `genre_labels` and `mood_labels` from the CURRENT tag
    partition, keyed on track_id.

    Stage 1 writes label vectors alongside the features, but labels depend only
    on annotations_final.csv and the tag lists -- not on audio. So whenever
    MIN_POSITIVES or the tag lists change, the cached stage 1 file is stale in
    its label columns while its features are still perfectly good. Recomputing
    them here costs seconds and means editing the tag partition never triggers a
    re-extraction. It also removes an entire class of silent bug: a label vector
    whose length no longer matches len(genre) + len(mood).

    Clips left with no positive target under the new partition are dropped --
    a narrower tag set can empty out a clip that previously qualified.
    """
    text_by_clip, genre_by_clip, mood_by_clip = {}, {}, {}
    for _, row in ann.iterrows():
        clip = int(row["clip_id"])
        text_by_clip[clip] = build_text(row, instrument)
        genre_by_clip[clip] = [int(row[t]) for t in genre]
        mood_by_clip[clip] = [int(row[t]) for t in mood]

    def relabel(batch):
        return {
            "text": [text_by_clip.get(int(c), "A music clip.")
                     for c in batch["track_id"]],
            "genre_labels": [genre_by_clip.get(int(c), [0] * len(genre))
                             for c in batch["track_id"]],
            "mood_labels": [mood_by_clip.get(int(c), [0] * len(mood))
                            for c in batch["track_id"]],
        }

    before = len(ds)
    ds = ds.map(relabel, batched=True, batch_size=1000, desc="relabel")
    ds = ds.filter(
        lambda b: [sum(g) + sum(m) > 0
                   for g, m in zip(b["genre_labels"], b["mood_labels"])],
        batched=True, batch_size=1000, desc="drop unlabelled",
    )

    if verbose:
        print(f"labels rebuilt for the current partition: "
              f"{len(genre)} genre + {len(mood)} mood")
        if len(ds) != before:
            print(f"  dropped {before - len(ds)} segments whose clips have no "
                  f"positive target under this tag set")
        row = ds[0]
        assert len(row["genre_labels"]) == len(genre)
        assert len(row["mood_labels"]) == len(mood)
        print(f"  example text: {row['text']!r}")
    return ds


def clip_index(ds) -> pd.DataFrame:
    """
    One row per clip: track_id, song_id, artist, genre_labels, mood_labels.
    Drops the frame-feature columns first so the split decision costs a few MB
    rather than touching the ~1.9 GB it does not need.
    """
    keep = ["track_id", "song_id", "artist", "genre_labels", "mood_labels"]
    slim = ds.remove_columns([c for c in ds.column_names if c not in keep])
    return slim.to_pandas().drop_duplicates("track_id")


def split_by_group(clips: pd.DataFrame, train_frac: float = TRAIN_FRAC,
                   val_frac: float = VAL_FRAC, seed: int = SEED,
                   group_by: str = GROUP_BY, verbose: bool = VERBOSE) -> dict:
    """
    Group-aware iterative multi-label stratified split.

    Keeps every clip from the same source group in one split while balancing
    the genre + mood label distribution across train / val / test.

    Why group at all: MTAT's clips are 29 s excerpts and ~5.8 of them come from
    each source recording, so a split that ignores the source routinely puts
    different slices of the SAME audio in train and test. A model can then
    score well by recognizing a mix rather than a genre.

        song   -> recommended; no recording spans two splits
        artist -> stricter; also removes "production sound" as a shortcut
        clip   -> no grouping, the standard folder protocol. Leaks.

    Each group gets ONE label vector, formed by taking the union (elementwise
    max) of its clips' labels -- not the first clip's. Different excerpts of one
    song carry different tags, so `.iloc[0]` would stratify on a single
    arbitrary sample and silently ignore the other ~5.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    key = {"song": "song_id", "artist": "artist", "clip": "track_id"}.get(group_by)
    if key is None:
        raise ValueError(f"GROUP_BY must be song/artist/clip, got {group_by!r}")
    if key not in clips.columns:
        raise KeyError(f"column {key!r} missing -- rebuild stage 1 so it is written")

    # one row per group: union of every clip's labels in that group
    groups, label_rows = [], []
    for group_id, df in clips.groupby(key, sort=True):
        genre = np.stack([np.asarray(v, dtype=np.int8) for v in df["genre_labels"]])
        mood = np.stack([np.asarray(v, dtype=np.int8) for v in df["mood_labels"]])
        groups.append(group_id)
        label_rows.append(np.concatenate([genre.max(axis=0), mood.max(axis=0)]))

    groups = np.asarray(groups, dtype=object)
    Y = np.stack(label_rows)
    X = np.zeros((len(groups), 1))

    # train vs (val + test)
    s1 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=1 - train_frac, random_state=seed)
    train_idx, temp_idx = next(s1.split(X, Y))

    # val vs test, splitting the remainder in proportion to val_frac
    test_frac = 1 - train_frac - val_frac
    s2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_frac / (val_frac + test_frac), random_state=seed)
    rel_val, rel_test = next(s2.split(np.zeros((len(temp_idx), 1)), Y[temp_idx]))

    assigned = {
        "train": set(groups[train_idx]),
        "val": set(groups[temp_idx[rel_val]]),
        "test": set(groups[temp_idx[rel_test]]),
    }
    ids = {s: set(clips[clips[key].isin(v)]["track_id"]) for s, v in assigned.items()}

    # a leak here is silent and fatal, so assert rather than trust
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (assigned[a] & assigned[b]), f"{group_by} leak: {a}/{b}"
        assert not (ids[a] & ids[b]), f"clip leak: {a}/{b}"
    covered = sum(len(v) for v in ids.values())
    assert covered == clips.track_id.nunique(), \
        f"{clips.track_id.nunique() - covered} clips fell out of every split"

    if verbose:
        n_clips = clips.track_id.nunique()
        print(f"\nstratified split (grouped by {group_by}):")
        for s in ("train", "val", "test"):
            part = clips[clips.track_id.isin(ids[s])]
            print(f"  {s:<6} {len(assigned[s]):>5} {group_by}s  "
                  f"{len(ids[s]):>6} clips ({len(ids[s]) / n_clips:5.1%})  "
                  f"{part.artist.nunique():>4} artists")
        if group_by == "song":
            shared = (set(clips[clips.track_id.isin(ids["train"])].artist) &
                      set(clips[clips.track_id.isin(ids["test"])].artist))
            print(f"  note: {len(shared)} artists appear in both train and test "
                  f"(expected -- grouping is by song). GROUP_BY='artist' removes this.")
        # groups are balanced by count; clips are not, since group sizes vary
        drift = max(abs(len(ids["train"]) / n_clips - train_frac),
                    abs(len(ids["val"]) / n_clips - val_frac))
        if drift > 0.03:
            print(f"  WARNING: clip proportions drift {drift:.1%} from target -- "
                  f"stratification balances GROUPS, and group sizes vary.")
    return ids


def write_split_parquets(split_ids: dict, ds, out_dir: Path = OUT_DIR,
                         verbose: bool = VERBOSE):
    """
    One batched filter pass per split. Arrow keeps the result memory-mapped, so
    the three files are produced without ever holding the dataset in RAM.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    for name, ids in split_ids.items():
        ids = set(ids)
        part = ds.filter(
            lambda batch: [t in ids for t in batch["track_id"]],
            batched=True,
            batch_size=1000,
            desc=f"stage 3: {name}",
        )
        path = out_dir / f"{name}.parquet"
        part.to_parquet(path)
        counts[name] = len(part)
        if verbose:
            print(f"  wrote {path}  ({len(part)} rows, "
                  f"{path.stat().st_size / 1e6:.1f} MB)")
    return counts


def report_label_balance(clips: pd.DataFrame, split_ids: dict,
                         genre, mood, verbose: bool = VERBOSE):
    """
    Checks whether the label stratification survived the grouping constraint.

    The split IS stratified on labels, but it stratifies GROUPS, not clips, and
    each group carries the union of its clips' labels. Since all ~5.8 clips of a
    song move together, a tag concentrated in a handful of recordings can still
    land unevenly. This is the only way to see whether that happened.

    Prints, per tag: positive CLIP counts in each split, and the share of that
    tag's positives that ended up in each split. Perfect stratification would
    put 70 / 15 / 15 percent everywhere; the `drift` column is how far the test
    share is from its target. A tag with near-zero test positives produces a
    meaningless per-tag F1 and drags Macro-F1 for reasons unrelated to the model.
    """
    names = list(genre) + list(mood)
    groups = ["genre"] * len(genre) + ["mood"] * len(mood)

    counts = {}
    for s, ids in split_ids.items():
        part = clips[clips.track_id.isin(ids)]
        vecs = []
        for col, expected in (("genre_labels", len(genre)), ("mood_labels", len(mood))):
            m = np.stack([np.asarray(v) for v in part[col]])
            if m.shape[1] != expected:
                raise ValueError(
                    f"{col} has {m.shape[1]} slots but the current partition has "
                    f"{expected} tags. The parquet was built with a different tag "
                    f"set -- attach_labels() should have fixed this, so check it "
                    f"ran before report_label_balance().")
            vecs.append(m.sum(axis=0))
        counts[s] = np.concatenate(vecs)

    total = counts["train"] + counts["val"] + counts["test"]
    safe = np.clip(total, 1, None)
    n_clips = clips.track_id.nunique()
    target = {s: len(ids) / n_clips for s, ids in split_ids.items()}

    table = pd.DataFrame({
        "group": groups,
        "train": counts["train"], "val": counts["val"], "test": counts["test"],
        "train%": (counts["train"] / safe * 100).round(1),
        "val%": (counts["val"] / safe * 100).round(1),
        "test%": (counts["test"] / safe * 100).round(1),
        "drift": (np.abs(counts["test"] / safe - target["test"]) * 100).round(1),
    }, index=names).sort_values(["group", "drift"], ascending=[True, False])

    if verbose:
        print(f"\nlabel balance across splits "
              f"(target {target['train']:.0%}/{target['val']:.0%}/{target['test']:.0%}, "
              f"positive CLIPS per tag):")
        print(table.to_string())
        print(f"\n  mean |test-share drift|  {table.drift.mean():.1f} pp"
              f"   |  worst {table.drift.max():.1f} pp ({table.drift.idxmax()})")

        thin = table[(table.val < 5) | (table.test < 5)]
        if len(thin):
            print(f"  WARNING: {len(thin)} tag(s) with <5 positives in val or test: "
                  f"{list(thin.index)}")
        dead = table[(table.train == 0) | (table.test == 0)]
        if len(dead):
            print(f"  WARNING: {len(dead)} tag(s) absent from train or test entirely: "
                  f"{list(dead.index)}")
        if table.drift.mean() > 8:
            print("  Stratification is struggling against the song grouping. Options: "
                  "raise MIN_POSITIVES, or try a different SEED and keep the split "
                  "with the lowest mean drift.")
    return table


def write_label_space(genre, mood, instrument, clips=None, split_ids=None,
                      out_dir: Path = OUT_DIR, verbose: bool = VERBOSE):
    """
    genre_labels and mood_labels are bare multi-hot vectors; this file is the
    only record of which position means which tag. mtat_graphs.py reads it.

    Also stores the TRAIN positive count for each target tag. MTAT is severely
    imbalanced -- classical has ~4,300 positives against disco's ~57 -- so
    BCEWithLogitsLoss needs pos_weight to stop the model predicting all-zeros
    for every rare tag. Saving the counts here means the model can build
    pos_weight without a second pass over the data, and guarantees the weights
    come from the train split only:

        counts = np.array(label_space["train_positives"])
        n = label_space["n_train_clips"]
        pos_weight = torch.tensor((n - counts) / np.clip(counts, 1, None),
                                  dtype=torch.float)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {"genre": genre, "mood": mood, "instrument": instrument,
               "n_genre": len(genre), "n_mood": len(mood)}

    if clips is not None and split_ids is not None:
        train = clips[clips.track_id.isin(split_ids["train"])]
        g = np.stack([np.asarray(v) for v in train["genre_labels"]]).sum(axis=0)
        m = np.stack([np.asarray(v) for v in train["mood_labels"]]).sum(axis=0)
        payload["train_positives"] = [int(v) for v in np.concatenate([g, m])]
        payload["n_train_clips"] = int(len(train))

    path = out_dir / "label_space.json"
    path.write_text(json.dumps(payload, indent=2))
    if verbose:
        print(f"  wrote {path}  ({len(genre)} genre + {len(mood)} mood tags"
              + (", with train positive counts for pos_weight"
                 if "train_positives" in payload else "") + ")")
    return path


# ===========================================================================
# 10. Main
# ===========================================================================

def main():
    print(f"{SEGMENT_SECONDS} s segments @ {SAMPLE_RATE} Hz | "
          f"chroma n_chroma={N_CHROMA}, mfcc n_mfcc={N_MFCC}, "
          f"n_fft={N_FFT}, hop_length={HOP_LENGTH}")
    print(f"batched map: batch_size={BATCH_SIZE}, "
          f"writer_batch_size={WRITER_BATCH_SIZE}, num_proc={NUM_PROC}\n")

    ann, info = load_annotations()
    instrument, genre, mood = partition_tags(ann)
    if not mood:
        raise SystemExit("no mood tags survived MIN_POSITIVES -- stopping. "
                         "Lower the threshold or switch datasets.")

    ensure_audio()

    build_segmented_dataset(ann, info, instrument, genre, mood)   # stage 1
    build_pooled_dataset()                                        # stage 2

    ds = load_pooled()                                            # stage 3
    ds = attach_labels(ds, ann, instrument, genre, mood)
    clips = clip_index(ds)
    split_ids = split_by_group(clips)
    report_label_balance(clips, split_ids, genre, mood)
    print()
    counts = write_split_parquets(split_ids, ds)
    write_label_space(genre, mood, instrument, clips, split_ids)

    print(f"\nsaved dataset versions in {OUT_DIR}:")
    for path in (SEGMENTED_PARQUET, POOLED_PARQUET,
                 OUT_DIR / "train.parquet", OUT_DIR / "val.parquet",
                 OUT_DIR / "test.parquet"):
        if path.exists():
            print(f"  {path.name:<38} {path.stat().st_size / 1e6:>8.1f} MB")

    print(f"\ndone. next: run src/mtat_graphs.py, which reads {OUT_DIR}/*.parquet")
    return counts


if __name__ == "__main__":
    main()