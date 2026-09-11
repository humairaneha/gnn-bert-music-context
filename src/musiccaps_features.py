"""
musiccaps_features.py -- Task 4 preprocessing, start to finish, in one file.

Same shape as mtat_features.py -- same constant names, same function names, same
staged parquet outputs -- joining two sources:

    CLAPv2/MusicCaps                    audio + caption, 5,352 clips, 9.83 GB
    humairaneha/MusicCaps-Curated-Tags  your 216 curated tags

joined on `ytid`, which is MusicCaps' primary key and is inherited by both.

Why MusicCaps for Task 4 and not MTAT: contrastive retrieval needs each caption
to identify one clip. MTAT has no captions, and the templated instrument
sentences used in Task 3 repeat across hundreds of clips, which caps R@1 below 1
no matter how good the model is. MusicCaps captions are expert-written free text,
one per clip. This is also the first point in the project where BERT's
pretraining is genuinely exercised rather than acting as a lookup over a small
tag vocabulary.

Pipeline:
    1. Join audio to tags on ytid (falls back to caption text if ytid is absent
       from the mirror). Report how many tagged clips actually have audio.
    2. Resample to 22,050 Hz, peak-normalize, cut each 10 s clip into 2 s
       non-overlapping segments -> 5 segments per clip.
    3. Per segment: chroma_stft (12 x 87) and MFCC (13 x 87), pooled to
       chroma_pooled (24-d) and mfcc_pooled (26-d).
    4. Multi-label stratified 70/15/15 split. NO grouping: MusicCaps clips come
       from distinct YouTube videos, so there is no shared-recording problem of
       the kind that forced song-grouping on MTAT.
    5. Write every stage to parquet, same as before.

The curated repo has three configs. `main` holds the metadata, caption and 216
bare one-hot tag columns; `tag_vocabulary` holds the tag -> semantic category
mapping (Genre, Instrument, Mood, Vocal, Tempo). Targets are taken from Genre
and Mood only.

Note on the join: the audio mirror has no ytid column, and the curated CSV's
ytid is partly corrupted -- IDs beginning with "-" were mangled to "#NAME?" by a
spreadsheet round-trip. The caption text is therefore the safer key, and since
MusicCaps captions are unique per clip it is a valid one. build_clip_table()
asserts that uniqueness rather than assuming it.

Run inspect_source() FIRST. It streams one row from each dataset, prints the
columns and sample rate, and costs nothing. The 9.83 GB download only starts
when you call main().

    from musiccaps_features import inspect_source
    inspect_source()

then

    python src/musiccaps_features.py
"""

from __future__ import annotations

import json
import warnings
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
AUDIO_REPO = "CLAPv2/MusicCaps"                      # pre-downloaded audio mirror
TAGS_REPO = "humairaneha/MusicCaps-Curated-Tags"     # your curated tags
OUT_DIR = Path("data/processed/musiccaps")

SEGMENTED_PARQUET = OUT_DIR / "segmented_audio_data.parquet"
POOLED_PARQUET = OUT_DIR / "segmented_audio_data_pooled.parquet"

# --- batching ---------------------------------------------------------------
BATCH_SIZE = 32
WRITER_BATCH_SIZE = 200
NUM_PROC = 1              # librosa + fork can be flaky on macOS; try 4, drop back to 1
USE_EXPLICIT_FEATURES = True

MAX_CLIPS = None          # cap for a fast development pass; try 200 first

# --- audio ------------------------------------------------------------------
# MusicCaps clips are 10 s. At 2 s windows that gives 5 segments -- the same node
# count as the MTAT graphs, so Task 3 and Task 4 graphs stay directly
# comparable. 22,050 Hz keeps the features comparable with Tasks 2 and 3; if the
# mirror stores audio at 48 kHz, librosa resamples down on load.
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 2                        # 10 s clip -> 5 segments
PEAK_NORMALIZE = True
MEL_FMAX = None                            # full-bandwidth source, no cap needed

# --- features ---------------------------------------------------------------
# At 22,050 Hz a 2 s segment is 44,100 samples, so hop 512 gives
# 1 + 44100//512 = 87 frames per segment.
N_CHROMA = 12                              # -> chroma_pooled is 24-d
N_MFCC = 13                                # -> mfcc_pooled is 26-d
N_FFT = 2048
HOP_LENGTH = 512
STD_DDOF = 1                               # matches torch.std, as in every other file

# --- what to store ----------------------------------------------------------
# 5,352 clips x 5 segments = ~26.8k rows. Frame features are ~230 MB; waveforms
# would add ~4.7 GB and nothing in Task 4 reads them.
KEEP_FRAME_FEATURES = True
KEEP_WAVEFORM = False

# --- label space ------------------------------------------------------------
# The curated repo stores tags as BARE one-hot column names ("rock", "ambient",
# "medium tempo") in the `main` config, and the tag -> semantic category mapping
# in a separate `tag_vocabulary` config. Five categories exist:
#     Genre, Instrument, Mood, Vocal, Tempo
# Instrument and Vocal are deliberately excluded from the targets -- they are
# the closest thing this dataset has to an input-side description, and keeping
# them out preserves the same separation used on MTAT.
VOCAB_CONFIG = "tag_vocabulary"
# Matched case-insensitively, since the README writes "Genre" and the CSV
# stores "genre".
TARGET_CATEGORIES = ["genre", "mood"]      # tempo/instrument/vocal not targets
MIN_POSITIVES = 55                         # same floor as the MTAT target tags

# --- splits -----------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42

VERBOSE = True

warnings.filterwarnings(
    "ignore", message="Trying to estimate tuning from empty frequency set")


# ===========================================================================
# 1. Inspect before downloading
# ===========================================================================

def inspect_source(audio_repo=AUDIO_REPO, tags_repo=TAGS_REPO):
    """
    Streams ONE row from the audio mirror and loads the tag metadata, then
    prints what is actually there. Costs seconds, not 9.83 GB.

    Run this before main(). The audio repo is a community mirror with no dataset
    card, so the column names are worth confirming rather than assuming.
    """
    print(f"--- {audio_repo} (streaming one row) ---")
    row = next(iter(load_dataset(audio_repo, split="train", streaming=True)))
    print(f"columns: {list(row.keys())}")
    a = row["audio"]
    print(f"audio: sr={a['sampling_rate']}, samples={len(a['array'])}, "
          f"duration={len(a['array']) / a['sampling_rate']:.1f}s")
    for k in ("ytid", "youtube_id", "id"):
        if k in row:
            print(f"join key present: {k!r} = {row[k]!r}")
            break
    else:
        print("NO ytid-like column -- the join will fall back to caption text")
    if "caption" in row:
        print(f"caption: {row['caption'][:110]}...")

    print(f"\n--- {tags_repo} ---")
    tags = load_dataset(tags_repo, "main", split="train")
    print(f"main: {len(tags)} rows, {len(tags.column_names)} columns")
    print(f"  metadata columns: {tags.column_names[:11]}")
    print(f"  first tag columns: {tags.column_names[11:16]} ...")

    cat_of = load_tag_categories(tags_repo)
    counts = Counter(cat_of.values())
    print(f"{VOCAB_CONFIG}: {len(cat_of)} tags across {len(counts)} categories")
    for cat, n in counts.most_common():
        mark = "  <- target" if cat in TARGET_CATEGORIES else ""
        print(f"  {cat:<12} {n:>4}{mark}")
    missing = [t for t in cat_of if t not in tags.column_names]
    if missing:
        print(f"  WARNING {len(missing)} vocabulary tags have no column: {missing[:5]}")
    return row, tags


def load_tag_categories(tags_repo=TAGS_REPO, config=VOCAB_CONFIG, verbose=VERBOSE):
    """
    Reads the tag -> semantic category mapping from the `tag_vocabulary` config.

    Columns are identified by CARDINALITY, not by name. With 216 tags and 5
    categories the distinction is unambiguous: the tag column has many distinct
    values, the category column has few. Matching on names like "tag" or
    "category" silently picks the wrong column when neither matches and the
    fallback grabs whichever column happens to come first.
    """
    vocab = load_dataset(tags_repo, config, split="train")
    cols = vocab.column_names
    distinct = {c: len(set(str(v) for v in vocab[c])) for c in cols}

    tag_col = max(cols, key=lambda c: distinct[c])          # ~216 distinct
    cat_col = min((c for c in cols if c != tag_col),        # ~5 distinct
                  key=lambda c: distinct[c])

    if distinct[tag_col] == distinct[cat_col]:
        raise ValueError(
            f"cannot tell the tag column from the category column in {config!r}: "
            f"distinct counts {distinct}. Set them by hand.")

    mapping = {str(a): str(b) for a, b in zip(vocab[tag_col], vocab[cat_col])}
    if verbose:
        print(f"{config}: tag column {tag_col!r} ({distinct[tag_col]} distinct), "
              f"category column {cat_col!r} ({distinct[cat_col]} distinct)")
        print(f"  categories: {sorted(set(mapping.values()))}")
    return mapping


# ===========================================================================
# 2. Join audio to tags
# ===========================================================================

def normalize_caption(s: str) -> str:
    """Mirrors sometimes re-wrap whitespace; collapse it before matching."""
    return " ".join(str(s).split())


def resolve_join_key(audio_cols, tag_cols):
    for k in ("ytid", "youtube_id", "id"):
        if k in audio_cols and k in tag_cols:
            return k
    return None


def build_clip_table(audio_repo=AUDIO_REPO, tags_repo=TAGS_REPO,
                     max_clips=MAX_CLIPS, verbose=VERBOSE):
    """
    Returns (clips_dataset, target_tags, category_of_tag).

    One row per clip: ytid, caption, labels (multi-hot), audio.
    Only clips that have BOTH audio and curated tags survive -- roughly 3% of
    MusicCaps is unrecoverable from YouTube, so a few tagged clips will have no
    audio and are dropped here rather than failing later.
    """
    tags = load_dataset(tags_repo, "main", split="train")
    audio = load_dataset(audio_repo, split="train")
    category_of_all = load_tag_categories(tags_repo)

    key = resolve_join_key(audio.column_names, tags.column_names)
    if key is None:
        if "caption" not in audio.column_names or "caption" not in tags.column_names:
            raise KeyError(
                f"no shared join key. audio has {audio.column_names}, "
                f"tags have {tags.column_names[:6]}...")
        if verbose:
            print("no ytid in both -- joining on caption text instead")
        tag_index = {normalize_caption(r["caption"]): r for r in tags}
        if len(tag_index) != len(tags):
            raise ValueError(
                f"{len(tags) - len(tag_index)} duplicate captions in the tag set; "
                f"caption is not a safe join key here")
        audio_key = lambda r: normalize_caption(r["caption"])
    else:
        if verbose:
            print(f"joining on {key!r}")
        tag_index = {r[key]: r for r in tags}
        audio_key = lambda r: r[key]

    have = {audio_key(r) for r in audio.select_columns(
        [c for c in audio.column_names if c != "audio"])}
    matched = have & tag_index.keys()
    if verbose:
        print(f"audio {len(have)} | tagged {len(tag_index)} | matched {len(matched)}"
              f" ({len(matched) / max(len(tag_index), 1):.1%} of tagged clips "
              f"have audio)")
        print(f"tagged but no audio: {len(tag_index.keys() - have)}")

    # --- label space, computed on the clips we ACTUALLY have ---
    # A tag near the threshold can drop below it once the ~3% without audio are
    # removed, so the floor is applied after the join, not before.
    tag_cols = {
        cat: [t for t, c in category_of_all.items()
              if c.strip().lower() == cat.strip().lower() and t in tags.column_names]
        for cat in TARGET_CATEGORIES
    }
    present = sorted(set(category_of_all.values()))
    for cat, cs in tag_cols.items():
        if not cs:
            raise KeyError(
                f"no tags in category {cat!r}. Categories in the vocabulary: "
                f"{present}. Set TARGET_CATEGORIES to a subset of those.")
    counts = {c: sum(int(tag_index[k][c]) for k in matched)
              for cat in TARGET_CATEGORIES for c in tag_cols[cat]}
    target_tags, category_of_tag = [], {}
    for cat in TARGET_CATEGORIES:
        kept = [c for c in tag_cols[cat] if counts[c] >= MIN_POSITIVES]
        target_tags += kept
        category_of_tag.update({c: cat.strip().lower() for c in kept})
        if verbose:
            print(f"  {cat}: {len(kept)}/{len(tag_cols[cat])} tags clear "
                  f">= {MIN_POSITIVES} positives")
    if not target_tags:
        raise SystemExit("no target tags survived MIN_POSITIVES -- lower the floor "
                         "or check TARGET_CATEGORIES against inspect_source()")

    audio = audio.filter(lambda r: audio_key(r) in matched,
                         desc="keep matched clips")
    if max_clips:
        audio = audio.select(range(min(max_clips, len(audio))))

    def attach(batch):
        keys = ([batch[key][i] for i in range(len(batch[key]))] if key
                else [normalize_caption(c) for c in batch["caption"]])
        return {
            "ytid": [str(k) for k in keys],
            "labels": [[int(tag_index[k][t]) for t in target_tags] for k in keys],
        }

    clips = audio.map(attach, batched=True, batch_size=256, desc="attach labels")
    if verbose:
        print(f"\n{len(clips)} clips with audio + {len(target_tags)} targets")
        print(f"example caption: {clips[0]['caption'][:110]}...")
    return clips, target_tags, category_of_tag


# ===========================================================================
# 3. Segmentation + features  (identical rules to the other pipelines)
# ===========================================================================

def segment_waveform(y: np.ndarray, sr=SAMPLE_RATE, seconds=SEGMENT_SECONDS):
    """
    Non-overlapping windows; an incomplete final window is dropped rather than
    zero-padded, so every segment yields the same frame count and the pooled
    statistics average over an equal number of frames for every node.
    """
    window = seconds * sr
    return [y[i:i + window] for i in range(0, len(y), window)
            if len(y[i:i + window]) == window]


def extract_chroma(segment, sr=SAMPLE_RATE):
    """(12, 87) chroma_stft -- pitch-class energy, i.e. harmonic content."""
    return librosa.feature.chroma_stft(
        y=segment, sr=sr, n_chroma=N_CHROMA, n_fft=N_FFT,
        hop_length=HOP_LENGTH).astype(np.float32)


def extract_mfcc(segment, sr=SAMPLE_RATE):
    """(13, 87) MFCCs -- spectral envelope, i.e. timbre."""
    kw = {} if MEL_FMAX is None else {"fmax": MEL_FMAX}
    return librosa.feature.mfcc(
        y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, **kw).astype(np.float32)


def pool_mean_std(feature_array):
    """
    (n_bins, T) -> (2 * n_bins,) as [means..., stds...]. The std half separates a
    sustained pad from a busy passage that averages to the same place. ddof=1
    matches torch.std, keeping these values comparable across all four datasets.
    """
    arr = np.asarray(feature_array, dtype=np.float32)
    return np.concatenate([arr.mean(axis=1),
                           arr.std(axis=1, ddof=STD_DDOF)]).astype(np.float32)


def as_matrix(value):
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.stack([np.asarray(r, dtype=np.float32) for r in arr])
    return arr.astype(np.float32, copy=False)


# ===========================================================================
# 4. Schema
# ===========================================================================

def build_features(with_frames=None, with_waveform=None) -> Features:
    """Declared, not inferred -- inference can silently store float32 as float64."""
    with_frames = KEEP_FRAME_FEATURES if with_frames is None else with_frames
    with_waveform = KEEP_WAVEFORM if with_waveform is None else with_waveform
    schema = {
        "track_id": Value("string"),        # = ytid, the MusicCaps primary key
        "segment_id": Value("int32"),
        "text": Value("string"),            # the real caption, not a template
        "labels": Sequence(Value("int8")),
        "is_silent": Value("bool"),
    }
    if with_frames:
        schema["chroma_features"] = Sequence(Sequence(Value("float32")))   # (12, 87)
        schema["mfcc_features"] = Sequence(Sequence(Value("float32")))     # (13, 87)
    if with_waveform:
        schema["segment"] = Sequence(Value("float32"))
    return Features(schema)


def pooled_features(base: Features) -> Features:
    schema = dict(base)
    schema["chroma_pooled"] = Sequence(Value("float32"))   # 24-d
    schema["mfcc_pooled"] = Sequence(Value("float32"))     # 26-d
    return Features(schema)


# ===========================================================================
# 5. Stage 1 -- segmented_audio_data.parquet
# ===========================================================================

SILENT_RMS = 1e-4
DROP_SILENT = False


def segment_and_extract(batch):
    """
    Batched map, one clip in and several segments out -- the same flattening
    pattern as the GTZAN and MTAT pipelines.

    Resampling happens here rather than at load: the mirror's native rate is
    whatever it is, and librosa.resample gets everything onto SAMPLE_RATE so the
    frame count is exactly 87 per segment regardless of source.
    """
    out = {k: [] for k in ("track_id", "segment_id", "text", "labels", "is_silent")}
    if KEEP_FRAME_FEATURES:
        out["chroma_features"], out["mfcc_features"] = [], []
    if KEEP_WAVEFORM:
        out["segment"] = []

    for i, audio in enumerate(batch["audio"]):
        y = np.asarray(audio["array"], dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=0)
        if audio["sampling_rate"] != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=audio["sampling_rate"],
                                 target_sr=SAMPLE_RATE)
        if PEAK_NORMALIZE and np.any(y):
            y = librosa.util.normalize(y)

        for segment_id, seg in enumerate(segment_waveform(y)):
            silent = bool(np.sqrt(np.mean(seg ** 2)) < SILENT_RMS)
            if silent and DROP_SILENT:
                continue
            out["is_silent"].append(silent)
            out["track_id"].append(str(batch["ytid"][i]))
            out["segment_id"].append(segment_id)
            out["text"].append(str(batch["caption"][i]))
            out["labels"].append(list(batch["labels"][i]))
            if KEEP_FRAME_FEATURES:
                out["chroma_features"].append(extract_chroma(seg))
                out["mfcc_features"].append(extract_mfcc(seg))
            if KEEP_WAVEFORM:
                out["segment"].append(seg.astype(np.float32))
    return out


def build_segmented_dataset(clips, out_path=SEGMENTED_PARQUET, verbose=VERBOSE):
    out_path = Path(out_path)
    if out_path.exists():
        if verbose:
            print(f"stage 1: reusing {out_path} -- delete it to rebuild")
        return out_path

    seg = clips.map(segment_and_extract, batched=True, batch_size=BATCH_SIZE,
                    num_proc=NUM_PROC if NUM_PROC > 1 else None,
                    writer_batch_size=WRITER_BATCH_SIZE,
                    remove_columns=clips.column_names,
                    features=build_features() if USE_EXPLICIT_FEATURES else None,
                    desc="stage 1: segment + chroma/mfcc")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seg.to_parquet(out_path)

    if verbose:
        per = Counter(Counter(seg["track_id"]).values())
        n_silent = int(np.sum(seg["is_silent"]))
        print(f"\nstage 1: {len(seg)} segments from "
              f"{len(set(seg['track_id']))} clips "
              f"({n_silent} near-silent, {n_silent / max(len(seg), 1):.2%})")
        print(f"  segments per clip: {dict(sorted(per.items()))}")
        print(f"  {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ===========================================================================
# 6. Stage 2 -- pooling
# ===========================================================================

def add_pooled(batch):
    return {
        "chroma_pooled": [pool_mean_std(as_matrix(c)) for c in batch["chroma_features"]],
        "mfcc_pooled": [pool_mean_std(as_matrix(m)) for m in batch["mfcc_features"]],
    }


def build_pooled_dataset(in_path=SEGMENTED_PARQUET, out_path=POOLED_PARQUET,
                         verbose=VERBOSE):
    """
    Its own stage because pooling is the part you are most likely to revisit --
    a change here is a two-minute rerun over the saved frame features rather
    than another pass over 5,352 audio clips.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    if out_path.exists():
        if verbose:
            print(f"stage 2: reusing {out_path} -- delete it to rebuild")
        return out_path

    seg = load_dataset("parquet", data_files=str(in_path))["train"]
    pooled = seg.map(add_pooled, batched=True, batch_size=BATCH_SIZE,
                     num_proc=NUM_PROC if NUM_PROC > 1 else None,
                     writer_batch_size=WRITER_BATCH_SIZE,
                     features=pooled_features(seg.features) if USE_EXPLICIT_FEATURES
                     else None,
                     desc="stage 2: pooling")
    pooled.to_parquet(out_path)
    if verbose:
        r = pooled[0]
        print(f"\nstage 2: {len(pooled)} rows -> {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  chroma_features {as_matrix(r['chroma_features']).shape}, "
              f"mfcc_features {as_matrix(r['mfcc_features']).shape}")
        print(f"  chroma_pooled {len(r['chroma_pooled'])}-d, "
              f"mfcc_pooled {len(r['mfcc_pooled'])}-d")
    return out_path


# ===========================================================================
# 7. Stage 3 -- multi-label stratified split
# ===========================================================================

def clip_index(path=POOLED_PARQUET) -> pd.DataFrame:
    ds = load_dataset("parquet", data_files=str(Path(path)))["train"]
    keep = ["track_id", "labels"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    return ds.to_pandas().drop_duplicates("track_id")


def split_clips(clips: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC,
                seed=SEED, verbose=VERBOSE) -> dict:
    """
    Iterative multi-label stratified split, NO grouping.

    Unlike MTAT -- where ~5.8 clips came from each source recording and grouping
    was mandatory -- MusicCaps clips are 10 s excerpts from distinct YouTube
    videos, so there is no shared-audio leak to defend against. This is the same
    split strategy as Task 1, which keeps the two comparable.
    """
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    ids = clips["track_id"].to_numpy()
    Y = np.stack([np.asarray(v, dtype=np.int8) for v in clips["labels"]])
    X = np.zeros((len(ids), 1))

    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=1 - train_frac,
                                          random_state=seed)
    train_idx, temp_idx = next(s1.split(X, Y))
    test_frac = 1 - train_frac - val_frac
    s2 = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_frac / (val_frac + test_frac), random_state=seed)
    rel_val, rel_test = next(s2.split(np.zeros((len(temp_idx), 1)), Y[temp_idx]))

    out = {"train": set(ids[train_idx]), "val": set(ids[temp_idx[rel_val]]),
           "test": set(ids[temp_idx[rel_test]])}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (out[a] & out[b]), f"clip leak: {a}/{b}"
    assert sum(len(v) for v in out.values()) == len(ids), "clips lost in the split"

    if verbose:
        print("\nstratified split (no grouping needed -- distinct source videos):")
        for s in ("train", "val", "test"):
            print(f"  {s:<6} {len(out[s]):>6} clips ({len(out[s]) / len(ids):5.1%})")
    return out


def report_label_balance(clips, split_ids, target_tags, category_of_tag,
                         verbose=VERBOSE):
    cols = {}
    for s, ids in split_ids.items():
        part = clips[clips.track_id.isin(ids)]
        m = np.stack([np.asarray(v) for v in part["labels"]])
        cols[s] = pd.Series(m.sum(axis=0), index=target_tags)
    table = pd.DataFrame(cols).fillna(0).astype(int)
    table.insert(0, "category", [category_of_tag[t] for t in target_tags])
    if verbose:
        print("\npositive clips per tag:")
        print(table.to_string())
        thin = table[(table["test"] < 5) | (table["val"] < 5)]
        if len(thin):
            print(f"\nWARNING: {len(thin)} tag(s) with <5 positives in val or test: "
                  f"{list(thin.index)}")
    return table


def write_split_parquets(split_ids, in_path=POOLED_PARQUET, out_dir=OUT_DIR,
                         verbose=VERBOSE):
    ds = load_dataset("parquet", data_files=str(Path(in_path)))["train"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, ids in split_ids.items():
        ids = set(ids)
        part = ds.filter(lambda b: [t in ids for t in b["track_id"]],
                         batched=True, batch_size=1000, desc=f"stage 3: {name}")
        path = Path(out_dir) / f"{name}.parquet"
        part.to_parquet(path)
        counts[name] = len(part)
        if verbose:
            print(f"  wrote {path}  ({len(part)} rows, "
                  f"{path.stat().st_size / 1e6:.1f} MB)")
    return counts


def write_label_space(target_tags, category_of_tag, clips, split_ids,
                      out_dir=OUT_DIR, verbose=VERBOSE):
    """
    The multi-hot `labels` column is positional; this file is the only record of
    which slot means which tag. Train positive counts are stored too, so a model
    can build pos_weight from train statistics alone without a second pass.
    """
    genre = [t for t in target_tags if category_of_tag[t] == "genre"]
    mood = [t for t in target_tags if category_of_tag[t] == "mood"]
    train = clips[clips.track_id.isin(split_ids["train"])]
    counts = np.stack([np.asarray(v) for v in train["labels"]]).sum(axis=0)

    payload = {"tags": target_tags, "genre": genre, "mood": mood,
               "n_genre": len(genre), "n_mood": len(mood),
               "category_of_tag": category_of_tag,
               "train_positives": [int(v) for v in counts],
               "n_train_clips": int(len(train))}
    path = Path(out_dir) / "label_space.json"
    path.write_text(json.dumps(payload, indent=2))
    if verbose:
        print(f"  wrote {path}  ({len(genre)} genre + {len(mood)} mood tags)")
    return path


# ===========================================================================
# 8. Main
# ===========================================================================

def main():
    print(f"{SEGMENT_SECONDS} s segments @ {SAMPLE_RATE} Hz | "
          f"chroma n_chroma={N_CHROMA}, mfcc n_mfcc={N_MFCC}, "
          f"n_fft={N_FFT}, hop_length={HOP_LENGTH}\n")

    clips_ds, target_tags, category_of_tag = build_clip_table()
    build_segmented_dataset(clips_ds)          # stage 1
    build_pooled_dataset()                     # stage 2

    clips = clip_index()                       # stage 3
    split_ids = split_clips(clips)
    report_label_balance(clips, split_ids, target_tags, category_of_tag)
    print()
    write_split_parquets(split_ids)
    write_label_space(target_tags, category_of_tag, clips, split_ids)

    print(f"\nsaved dataset versions in {OUT_DIR}:")
    for p in (SEGMENTED_PARQUET, POOLED_PARQUET, OUT_DIR / "train.parquet",
              OUT_DIR / "val.parquet", OUT_DIR / "test.parquet"):
        if p.exists():
            print(f"  {p.name:<38} {p.stat().st_size / 1e6:>8.1f} MB")
    print(f"\ndone. next: build graphs, then run task4_contrastive.py")


if __name__ == "__main__":
    main()