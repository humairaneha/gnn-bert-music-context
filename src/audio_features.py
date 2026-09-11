"""Dispatch dataset preprocessing from the repository root.

    python src/audio_features.py --dataset mtat
    python src/audio_features.py --dataset musiccaps
    python src/audio_features.py --dataset gtzan

MTAT and MusicCaps extract segment features and write parquet splits.
The current GTZAN target validates existing parquet splits and writes label
metadata; it requires data/processed/GTZAN/{train,val,test}.parquet.
Dataset-specific settings are defined in each target module.
"""
import argparse
import runpy
from pathlib import Path

HERE = Path(__file__).parent

# Filenames are matched EXACTLY, including case. macOS is case-insensitive by
# default so a mismatch works locally and then fails on a Linux checkout, which
# is the worst possible time to find out.
TARGETS = {
    "gtzan":     "GTZAN_features.py",
    "mtat":      "MTAT_features.py",
    "musiccaps": "musiccaps_features.py",
}


def resolve(dataset: str) -> Path:
    path = HERE / TARGETS[dataset]
    if not path.exists():
        available = sorted(p.name for p in HERE.glob("*features*.py"))
        raise SystemExit(
            f"{path.name} not found in {HERE}.\n"
            f"Files matching '*features*': {available}\n"
            f"Rename the file or edit TARGETS in {Path(__file__).name}.")
    return path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(TARGETS), required=True)
    a = p.parse_args()
    script = resolve(a.dataset)
    print(f"[audio_features] {a.dataset} -> {script.name}\n")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
