"""Validate prepared dataset splits and write label_space.json.

The default input directory is data/processed/GTZAN/. It must contain
train.parquet, val.parquet, and test.parquet. This utility reads label and ID
columns, checks split disjointness, and records the target order and counts.
It does not extract audio features.

GTZAN label names come from the genre column. MTAT and MusicCaps use positional
multi-hot labels, so their names require matching metadata or the configured
fallback vocabulary. When changing the target vocabulary, update METADATA_PATH
or retain the matching label_space.json. Vector lengths are validated.

    python src/GTZAN_features.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "pyarrow is required. Install it with: pip install pyarrow"
    ) from exc


# ============================================================================
# CONFIGURATION -- EDIT ONLY THIS BLOCK
# ============================================================================

# Folder containing:
#   train.parquet
#   val.parquet
#   test.parquet
DATA_DIR = Path("data/processed/GTZAN")

# Usually leave this as "auto".
# The script detects:
#   GTZAN     -> genre
#   MTAT      -> genre_labels + mood_labels
#   MusicCaps -> labels
DATASET = "auto"

# Optional.
# For MTAT/MusicCaps, if an existing correct label_space.json is already
# inside DATA_DIR, leave this as None.
# Otherwise set its path here, e.g.:
# METADATA_PATH = Path("data/processed_mtat/old_label_space.json")
METADATA_PATH = None

OUTPUT_NAME = "label_space.json"
VERBOSE = True


# ============================================================================
# Current semantic-vocabulary fallbacks
# ============================================================================

MTAT_FALLBACK = {'genre': ['classical', 'baroque', 'opera', 'medieval', 'rock', 'hard rock', 'heavy metal', 'metal', 'punk', 'pop', 'dance', 'house', 'trance', 'techno', 'electronic', 'electro', 'industrial', 'jazz', 'blues', 'funky', 'country', 'folk', 'celtic', 'irish', 'spanish', 'ambient', 'new age', 'rap', 'indian', 'india', 'eastern', 'middle eastern', 'oriental', 'arabic', 'tribal', 'foreign'], 'mood': ['loud', 'quiet', 'soft', 'hard', 'heavy', 'slow', 'fast', 'fast beat', 'upbeat', 'dark', 'calm', 'mellow', 'sad', 'spacey', 'weird', 'strange', 'drone'], 'instrument': ['guitar', 'guitars', 'electric guitar', 'classical guitar', 'acoustic', 'acoustic guitar', 'no strings', 'banjo', 'sitar', 'harp', 'lute', 'fiddle', 'plucking', 'violin', 'violins', 'cello', 'strings', 'string', 'no violin', 'no guitar', 'piano', 'no piano', 'piano solo', 'harpsichord', 'harpsicord', 'organ', 'keyboard', 'synth', 'synthesizer', 'electric', 'flute', 'flutes', 'no flute', 'oboe', 'clarinet', 'sax', 'trumpet', 'horn', 'horns', 'wind', 'drums', 'drum', 'no drums', 'percussion', 'bongos', 'clapping', 'beat', 'beats', 'no beat', 'bass', 'bells', 'chimes', 'orchestra', 'orchestral', 'instrumental', 'solo', 'water', 'birds', 'vocal', 'vocals', 'no vocals', 'no vocal', 'voice', 'no voice', 'voices', 'singer', 'singing', 'no singing', 'no singer', 'male singer', 'female singer', 'man singing', 'woman singing', 'female singing', 'male vocal', 'female vocal', 'male voice', 'female voice', 'male vocals', 'female vocals', 'male', 'female', 'man', 'woman', 'men', 'women', 'girl', 'choir', 'choral', 'chorus', 'chant', 'chanting', 'monks', 'duet', 'operatic', 'male opera', 'female opera', 'talking']}

MUSICCAPS_FALLBACK = {'tags': ['ambient', 'arcade', 'ballad', 'band', 'blues', "children's music", 'cinematic', 'classical', 'country', 'dance', 'edm', 'electronic', 'folk', 'funk', 'gospel', 'groove', 'hip hop', 'indian', 'indie', 'jazz', 'latin', 'love', 'metal', 'minimalist', 'orchestral', 'pop', 'r&b', 'rap', 'reggae', 'religious', 'retro', 'rock', 'techno', 'trance', 'calm', 'creepy', 'dreamy', 'emotional', 'energetic', 'epic', 'exciting', 'fun', 'happy', 'haunting', 'hypnotic', 'inspiring', 'meditative', 'mysterious', 'ominous', 'psychedelic', 'romantic', 'sad', 'spiritual', 'suspenseful', 'tense'], 'genre': ['ambient', 'arcade', 'ballad', 'band', 'blues', "children's music", 'cinematic', 'classical', 'country', 'dance', 'edm', 'electronic', 'folk', 'funk', 'gospel', 'groove', 'hip hop', 'indian', 'indie', 'jazz', 'latin', 'love', 'metal', 'minimalist', 'orchestral', 'pop', 'r&b', 'rap', 'reggae', 'religious', 'retro', 'rock', 'techno', 'trance'], 'mood': ['calm', 'creepy', 'dreamy', 'emotional', 'energetic', 'epic', 'exciting', 'fun', 'happy', 'haunting', 'hypnotic', 'inspiring', 'meditative', 'mysterious', 'ominous', 'psychedelic', 'romantic', 'sad', 'spiritual', 'suspenseful', 'tense'], 'instrument': [], 'category_of_tag': {'ambient': 'genre', 'arcade': 'genre', 'ballad': 'genre', 'band': 'genre', 'blues': 'genre', "children's music": 'genre', 'cinematic': 'genre', 'classical': 'genre', 'country': 'genre', 'dance': 'genre', 'edm': 'genre', 'electronic': 'genre', 'folk': 'genre', 'funk': 'genre', 'gospel': 'genre', 'groove': 'genre', 'hip hop': 'genre', 'indian': 'genre', 'indie': 'genre', 'jazz': 'genre', 'latin': 'genre', 'love': 'genre', 'metal': 'genre', 'minimalist': 'genre', 'orchestral': 'genre', 'pop': 'genre', 'r&b': 'genre', 'rap': 'genre', 'reggae': 'genre', 'religious': 'genre', 'retro': 'genre', 'rock': 'genre', 'techno': 'genre', 'trance': 'genre', 'calm': 'mood', 'creepy': 'mood', 'dreamy': 'mood', 'emotional': 'mood', 'energetic': 'mood', 'epic': 'mood', 'exciting': 'mood', 'fun': 'mood', 'happy': 'mood', 'haunting': 'mood', 'hypnotic': 'mood', 'inspiring': 'mood', 'meditative': 'mood', 'mysterious': 'mood', 'ominous': 'mood', 'psychedelic': 'mood', 'romantic': 'mood', 'sad': 'mood', 'spiritual': 'mood', 'suspenseful': 'mood', 'tense': 'mood'}}


# ============================================================================
# Small helpers
# ============================================================================

def parquet_columns(path: Path) -> list[str]:
    """Read parquet schema without loading the large feature arrays."""
    return list(pq.ParquetFile(path).schema_arrow.names)


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only columns needed for label-space construction."""
    available = set(parquet_columns(path))
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(
            f"{path} is missing required column(s): {missing}\n"
            f"Available columns: {sorted(available)}"
        )
    return pd.read_parquet(path, columns=columns)


def to_python(value: Any) -> Any:
    """Convert numpy scalar values to normal JSON-serializable Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def sorted_labels(values) -> list:
    values = [to_python(v) for v in values]
    if not values:
        return []

    # Preserve natural numeric ordering for integer/float encoded classes.
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return sorted(values)

    return sorted(values, key=lambda x: str(x).lower())


def vector(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64).reshape(-1)
    return arr


def vector_key(value) -> tuple:
    return tuple(vector(value).tolist())


def assert_binary_matrix(mat: np.ndarray, name: str):
    unique = np.unique(mat)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(
            f"{name} must contain binary multi-hot values 0/1, "
            f"but found: {unique[:20].tolist()}"
        )


def unique_clip_rows(df: pd.DataFrame, label_columns: list[str], split_name: str):
    """
    Return one row per track_id and verify that every segment from a track has
    exactly the same label(s).
    """
    work = df.copy()

    for col in label_columns:
        if col in work.columns and len(work):
            first = work[col].iloc[0]
            if isinstance(first, (list, tuple, np.ndarray)):
                work[f"__{col}_key"] = work[col].map(vector_key)
            else:
                work[f"__{col}_key"] = work[col].map(to_python)

    key_cols = [f"__{c}_key" for c in label_columns]

    if key_cols:
        inconsistent = []
        for track_id, group in work.groupby("track_id", sort=False):
            for key_col in key_cols:
                if group[key_col].nunique(dropna=False) != 1:
                    inconsistent.append(track_id)
                    break

        if inconsistent:
            raise ValueError(
                f"{split_name} contains {len(inconsistent)} track(s) whose "
                f"segments do not share the same labels. Examples: "
                f"{inconsistent[:10]}"
            )

    return df.drop_duplicates("track_id").reset_index(drop=True)


def validate_split_disjointness(train, val, test):
    ids = {
        "train": set(train["track_id"].astype(str)),
        "val": set(val["track_id"].astype(str)),
        "test": set(test["track_id"].astype(str)),
    }

    overlaps = {
        "train/val": ids["train"] & ids["val"],
        "train/test": ids["train"] & ids["test"],
        "val/test": ids["val"] & ids["test"],
    }

    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        msg = ", ".join(f"{k}={len(v)}" for k, v in bad.items())
        raise ValueError(
            "track_id leakage detected across dataset splits: " + msg
        )


def load_metadata(metadata_path: Path | None, output_path: Path, dataset: str) -> dict:
    """
    Metadata priority:
      1. explicit --metadata file
      2. existing label_space.json in the dataset directory
      3. embedded current fallback for MTAT/MusicCaps
    """
    candidate = None

    if metadata_path is not None:
        candidate = metadata_path
    elif output_path.exists():
        candidate = output_path

    if candidate is not None and candidate.exists():
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)

    if dataset == "mtat":
        return MTAT_FALLBACK.copy()
    if dataset == "musiccaps":
        return MUSICCAPS_FALLBACK.copy()
    return {}


def category_map(tags, genre, mood):
    g = set(genre)
    m = set(mood)
    result = {}
    for tag in tags:
        key = str(tag)
        if tag in g:
            result[key] = "genre"
        elif tag in m:
            result[key] = "mood"
        else:
            raise ValueError(f"Target {tag!r} is in neither genre nor mood.")
    return result


# ============================================================================
# Dataset detection
# ============================================================================

def detect_dataset(columns: list[str]) -> str:
    cols = set(columns)

    if {"genre_labels", "mood_labels"}.issubset(cols):
        return "mtat"

    if "labels" in cols:
        return "musiccaps"

    if "genre" in cols:
        return "gtzan"

    raise ValueError(
        "Could not auto-detect dataset from train.parquet.\n"
        f"Columns found: {sorted(cols)}\n"
        "Expected one of:\n"
        "  GTZAN     -> genre\n"
        "  MTAT      -> genre_labels + mood_labels\n"
        "  MusicCaps -> labels"
    )


# ============================================================================
# GTZAN
# ============================================================================

def build_gtzan(train_path, val_path, test_path):
    required = ["track_id", "genre"]

    train = read_columns(train_path, required)
    val = read_columns(val_path, required)
    test = read_columns(test_path, required)

    validate_split_disjointness(train, val, test)

    # Validate consistency within each original track, but keep ALL rows for counting.
    unique_clip_rows(train, ["genre"], "train")
    unique_clip_rows(val, ["genre"], "val")
    unique_clip_rows(test, ["genre"], "test")

    all_genres = sorted_labels(
        pd.concat(
            [train["genre"], val["genre"], test["genre"]],
            ignore_index=True,
        ).dropna().unique().tolist()
    )

    train_genres = set(train["genre"].map(to_python))
    missing_from_train = [g for g in all_genres if g not in train_genres]
    if missing_from_train:
        raise ValueError(
            "GTZAN has class(es) present in val/test but absent from train: "
            f"{missing_from_train}"
        )

    counts = [
        int((train["genre"].map(to_python) == genre).sum())
        for genre in all_genres
    ]

    tags = all_genres
    mood = []
    instrument = []

    return {
        "tags": tags,
        "genre": all_genres,
        "mood": mood,
        "instrument": instrument,
        "n_genre": len(all_genres),
        "n_mood": 0,
        "n_instrument": 0,
        "category_of_tag": category_map(tags, all_genres, mood),
        "train_positives": counts,
        "n_train_clips": int(len(train)),
    }


# ============================================================================
# MTAT
# ============================================================================

def check_vector_length(df, col, expected, split_name):
    if len(df) == 0:
        return

    lengths = df[col].map(lambda x: len(vector(x))).unique().tolist()
    if lengths != [expected]:
        raise ValueError(
            f"{split_name}.{col} vector lengths are {lengths}, "
            f"but metadata expects {expected}."
        )


def build_mtat(train_path, val_path, test_path, metadata):
    required = ["track_id", "genre_labels", "mood_labels"]

    train = read_columns(train_path, required)
    val = read_columns(val_path, required)
    test = read_columns(test_path, required)

    validate_split_disjointness(train, val, test)

    genre = list(metadata.get("genre", []))
    mood = list(metadata.get("mood", []))
    instrument = list(metadata.get("instrument", []))

    if not genre and not mood:
        raise ValueError(
            "MTAT parquet stores positional vectors, not semantic tag names. "
            "Provide the correct old label_space.json with --metadata, or "
            "restore the embedded vocabulary."
        )

    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        check_vector_length(df, "genre_labels", len(genre), split_name)
        check_vector_length(df, "mood_labels", len(mood), split_name)

    # Validate consistency within each original track, but keep ALL rows for counting.
    unique_clip_rows(train, ["genre_labels", "mood_labels"], "train")
    unique_clip_rows(val, ["genre_labels", "mood_labels"], "val")
    unique_clip_rows(test, ["genre_labels", "mood_labels"], "test")

    if len(train):
        g = np.stack(train["genre_labels"].map(vector).to_list())
        m = np.stack(train["mood_labels"].map(vector).to_list())
    else:
        g = np.zeros((0, len(genre)), dtype=np.int64)
        m = np.zeros((0, len(mood)), dtype=np.int64)

    assert_binary_matrix(g, "MTAT genre_labels")
    assert_binary_matrix(m, "MTAT mood_labels")

    counts = np.concatenate(
        [g.sum(axis=0), m.sum(axis=0)]
    ).astype(int).tolist()

    tags = genre + mood

    return {
        "tags": tags,
        "genre": genre,
        "mood": mood,
        "instrument": instrument,
        "n_genre": len(genre),
        "n_mood": len(mood),
        "n_instrument": len(instrument),
        "category_of_tag": category_map(tags, genre, mood),
        "train_positives": counts,
        "n_train_clips": int(len(train)),
    }


# ============================================================================
# MusicCaps
# ============================================================================

def build_musiccaps(train_path, val_path, test_path, metadata):
    required = ["track_id", "labels"]

    train = read_columns(train_path, required)
    val = read_columns(val_path, required)
    test = read_columns(test_path, required)

    validate_split_disjointness(train, val, test)

    genre = list(metadata.get("genre", []))
    mood = list(metadata.get("mood", []))
    instrument = list(metadata.get("instrument", []))

    tags = list(metadata.get("tags", []))
    if not tags:
        tags = genre + mood

    if not tags:
        raise ValueError(
            "MusicCaps parquet stores a positional `labels` vector, not tag names. "
            "Provide the correct old label_space.json with --metadata, or "
            "restore the embedded vocabulary."
        )

    if genre + mood != tags:
        # Keep target ordering consistent across datasets.
        raise ValueError(
            "MusicCaps metadata ordering must satisfy tags == genre + mood. "
            "This is required so train_positives and model outputs have one "
            "unambiguous common ordering."
        )

    expected = len(tags)
    for split_name, df in [("train", train), ("val", val), ("test", test)]:
        check_vector_length(df, "labels", expected, split_name)

    # Validate consistency within each original track, but keep ALL rows for counting.
    unique_clip_rows(train, ["labels"], "train")
    unique_clip_rows(val, ["labels"], "val")
    unique_clip_rows(test, ["labels"], "test")

    if len(train):
        y = np.stack(train["labels"].map(vector).to_list())
    else:
        y = np.zeros((0, expected), dtype=np.int64)

    assert_binary_matrix(y, "MusicCaps labels")
    counts = y.sum(axis=0).astype(int).tolist()

    return {
        "tags": tags,
        "genre": genre,
        "mood": mood,
        "instrument": instrument,
        "n_genre": len(genre),
        "n_mood": len(mood),
        "n_instrument": len(instrument),
        "category_of_tag": category_map(tags, genre, mood),
        "train_positives": counts,
        "n_train_clips": int(len(train)),
    }


# ============================================================================
# Public entry point
# ============================================================================

def build_label_space(
    data_dir: str | Path,
    dataset: str = "auto",
    metadata_path: str | Path | None = None,
    output_name: str = "label_space.json",
    verbose: bool = True,
):
    data_dir = Path(data_dir)

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    output_path = data_dir / output_name

    for path in (train_path, val_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"Required split not found: {path}")

    detected = detect_dataset(parquet_columns(train_path))

    if dataset == "auto":
        dataset = detected
    else:
        dataset = dataset.lower()
        if dataset != detected:
            raise ValueError(
                f"--dataset={dataset!r}, but parquet schema looks like "
                f"{detected!r}."
            )

    metadata_path = Path(metadata_path) if metadata_path else None
    metadata = load_metadata(metadata_path, output_path, dataset)

    if dataset == "gtzan":
        payload = build_gtzan(train_path, val_path, test_path)
    elif dataset == "mtat":
        payload = build_mtat(train_path, val_path, test_path, metadata)
    elif dataset == "musiccaps":
        payload = build_musiccaps(train_path, val_path, test_path, metadata)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Universal sanity checks
    assert payload["tags"] == payload["genre"] + payload["mood"]
    assert len(payload["train_positives"]) == len(payload["tags"])
    assert payload["n_genre"] == len(payload["genre"])
    assert payload["n_mood"] == len(payload["mood"])
    assert payload["n_instrument"] == len(payload["instrument"])

    data_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if verbose:
        print("=" * 72)
        print(f"Dataset          : {dataset}")
        print(f"Data directory   : {data_dir}")
        print(f"Train rows       : {payload['n_train_clips']}")
        print(f"Genre targets    : {payload['n_genre']}")
        print(f"Mood targets     : {payload['n_mood']}")
        print(f"Total targets    : {len(payload['tags'])}")
        print(f"Instrument terms : {payload['n_instrument']}")
        print(f"Saved            : {output_path}")
        print("=" * 72)

        print("\nTarget ordering:")
        for i, (tag, count) in enumerate(
            zip(payload["tags"], payload["train_positives"])
        ):
            print(f"{i:3d}  {str(tag):<30} positives={count}")

    return payload


def main():
    return build_label_space(
        data_dir=DATA_DIR,
        dataset=DATASET,
        metadata_path=METADATA_PATH,
        output_name=OUTPUT_NAME,
        verbose=VERBOSE,
    )


if __name__ == "__main__":
    main()
