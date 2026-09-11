"""Run a task-specific training pipeline from the repository root.

Dataset and model settings are defined in the target module's configuration.

    python src/train.py --task 1
    python src/train.py --task 2
    python src/train.py --task 3
    python src/train.py --task 3 --variant mlp
    python src/train.py --task 3 --variant fusion
    python src/train.py --task 4
    python src/train.py --task 4 --variant supervised
"""
import argparse
import runpy
from pathlib import Path

HERE = Path(__file__).parent

TARGETS = {
    ("1", "default"):    "bert_musiccaps_task1.py",
    ("2", "default"):    "GTZAN_gnn.py",
    ("3", "default"):    "gnn.py",
    ("3", "mlp"):        "task3_mlp_nograph.py",
    ("3", "fusion"):     "task3_fusion.py",
    ("4", "default"):    "run_musiccaps_task4.py",
    ("4", "supervised"): "run_musiccaps_supervised.py",
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["1", "2", "3", "4"], required=True)
    p.add_argument("--variant", default="default",
                   choices=["default", "mlp", "fusion", "supervised"])
    a = p.parse_args()

    key = (a.task, a.variant)
    if key not in TARGETS:
        raise SystemExit(f"no such combination: task {a.task}, variant "
                         f"{a.variant!r}. Available: {sorted(TARGETS)}")
    script = HERE / TARGETS[key]
    if not script.exists():
        raise SystemExit(f"{script.name} not found in {HERE}. Check the target filename "
                         f"or edit TARGETS at the top of {Path(__file__).name}.")
    print(f"[train] task {a.task} ({a.variant}) -> {script.name}\n")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
