"""
graph_builder.py -- graph construction entry point.

Builds one PyG graph per clip from the preprocessed parquet files. Nodes are
time segments; edges are a temporal chain plus similarity edges between
non-adjacent segments, using track-centred cosine similarity.

    python src/graph_builder.py --dataset gtzan
    python src/graph_builder.py --dataset mtat
    python src/graph_builder.py --dataset musiccaps

Run the matching audio_features.py stage first.
"""
import argparse
import runpy
from pathlib import Path

HERE = Path(__file__).parent

TARGETS = {
    "gtzan":     "GTZAN_graphs.py",
    "mtat":      "MTAT_graphs.py",
    "musiccaps": "graphs.py",
}


def resolve(dataset: str) -> Path:
    path = HERE / TARGETS[dataset]
    if not path.exists():
        available = sorted(p.name for p in HERE.glob("*graph*.py"))
        raise SystemExit(
            f"{path.name} not found in {HERE}.\n"
            f"Files matching '*graph*': {available}\n"
            f"Rename the file or edit TARGETS in {Path(__file__).name}.")
    return path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(TARGETS), required=True)
    a = p.parse_args()
    script = resolve(a.dataset)
    print(f"[graph_builder] {a.dataset} -> {script.name}\n")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
