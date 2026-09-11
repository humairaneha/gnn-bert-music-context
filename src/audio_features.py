"""
audio_features.py -- preprocessing entry point.

Dispatches to the dataset-specific pipeline. Each of those is a single
config-block-driven script; this file exists to give the repository the
interface the project specification asks for.

    python src/audio_features.py --dataset gtzan
    python src/audio_features.py --dataset mtat
    python src/audio_features.py --dataset musiccaps

Every pipeline shares the same stages and writes the same columns:
    audio -> segments -> chroma/MFCC -> mean+std pooling -> stratified split
    -> data/processed/<dataset>/{train,val,test}.parquet
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
