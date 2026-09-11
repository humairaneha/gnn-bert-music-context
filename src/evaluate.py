"""
evaluate.py -- collect every run's metrics into results/metrics.json.

Evaluation itself happens inside each training script, because thresholds are
tuned on validation and applied to test within the same run; re-deriving them
here would risk a different protocol per row and silently break comparability.
This file walks the results tree instead, aggregates whatever each run wrote
into one file, and prints the comparison tables used in the report.

Runs write metrics in several shapes -- some nest under "test_tuned", some are
flat, some use "macro_auc_pr" instead of "auc_pr". Rather than hard-coding a
list of key names and silently mislabelling anything unrecognised, this searches
each file for the first dictionary containing a Macro-F1 and normalises the
aliases.

    python src/evaluate.py
    python src/evaluate.py --verbose     # show which key each run was read from
"""
import argparse
import json
from pathlib import Path

RESULTS = Path("results")

# Files a run might have written. test_metrics.json first so it wins when both
# exist in the same folder.
FILENAMES = ("test_metrics.json", "metrics.json")

# Preference order when a file nests several metric blocks.
PREFERRED = ("test_tuned", "test", "test_at_0.5", "zero_shot_tagging",
             "test_track_level", "val", "best_val")

# Different scripts named the same quantity differently.
ALIASES = {
    "auc_pr":         ("auc_pr", "macro_auc_pr", "mean_auc_pr", "AUC-PR"),
    "macro_f1":       ("macro_f1", "macro_F1", "best_val_macro_f1"),
    "micro_f1":       ("micro_f1", "micro_F1", "best_val_micro_f1"),
    "genre_macro_f1": ("genre_macro_f1", "genre_f1", "genre"),
    "mood_macro_f1":  ("mood_macro_f1", "mood_f1", "mood"),
}


def normalise(d: dict) -> dict:
    """Map whatever this run called things onto one vocabulary."""
    out = {}
    for canonical, names in ALIASES.items():
        for n in names:
            if isinstance(d.get(n), (int, float)):
                out[canonical] = float(d[n])
                break
    return out


def has_f1(d) -> bool:
    return isinstance(d, dict) and any(
        isinstance(d.get(n), (int, float)) for n in ALIASES["macro_f1"])


def row(m: dict):
    """
    Returns (normalised_metrics, source_key) or (None, None).

    Checks the preferred keys, then the top level, then any nested dict --
    so a run that invented its own key name is still picked up rather than
    reported as having no metrics.
    """
    for key in PREFERRED:
        if has_f1(m.get(key)):
            return normalise(m[key]), key
    if has_f1(m):
        return normalise(m), "(top level)"

    # Recursive descent, breadth-first, for arbitrarily nested shapes such as
    # {"gnn": {"mfcc_sage": {"test": {...}}}}. Preferred keys win at every
    # depth, so a file containing both "val" and "test" still reports "test".
    queue = [(k, v) for k, v in m.items() if isinstance(v, dict)]
    while queue:
        nxt = []
        for path, node in queue:
            for key in PREFERRED:
                if has_f1(node.get(key)):
                    return normalise(node[key]), f"{path}.{key}"
            if has_f1(node):
                return normalise(node), path
            nxt += [(f"{path}.{k}", v) for k, v in node.items()
                    if isinstance(v, dict)]
        queue = nxt
    return None, None


def collect(root: Path):
    """One entry per run directory; test_metrics.json wins over metrics.json."""
    runs = {}
    for name in FILENAMES:
        for path in sorted(root.rglob(name)):
            if path.parent == root and name == "metrics.json":
                continue                      # our own aggregate output
            key = str(path.parent.relative_to(root)) or path.stem
            runs.setdefault(key, json.loads(path.read_text()))
    return runs


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, default=RESULTS)
    p.add_argument("--verbose", action="store_true",
                   help="show which key each run's metrics were read from")
    a = p.parse_args()

    runs = collect(a.results_dir)
    if not runs:
        raise SystemExit(f"no metrics files found under {a.results_dir}")
    (a.results_dir / "metrics.json").write_text(json.dumps(runs, indent=2))

    def cell(r, k):
        return f"{r[k]:>10.4f}" if k in r else f"{'--':>10}"

    print(f"{'run':<40}{'Macro-F1':>10}{'Micro-F1':>10}{'AUC-PR':>10}"
          f"{'genre':>10}{'mood':>10}")
    print("-" * 90)
    unread = []
    for name, m in sorted(runs.items()):
        r, src = row(m)
        if r is None:
            if "caption_to_audio" not in m:
                unread.append((name, sorted(m)[:6]))
            continue
        print(f"{name:<40}{cell(r,'macro_f1')}{cell(r,'micro_f1')}"
              f"{cell(r,'auc_pr')}{cell(r,'genre_macro_f1')}"
              f"{cell(r,'mood_macro_f1')}"
              + (f"   [{src}]" if a.verbose else ""))

    ret = {k: v for k, v in runs.items() if "caption_to_audio" in v}
    if ret:
        print(f"\n{'retrieval run':<44}{'R@1':>9}{'R@5':>9}{'R@10':>9}")
        print("-" * 71)
        for name, m in sorted(ret.items()):
            for d in ("caption_to_audio", "audio_to_caption"):
                if d in m:
                    r = m[d]
                    print(f"{name + ' / ' + d:<44}{r['R@1']:>9.4f}"
                          f"{r['R@5']:>9.4f}{r['R@10']:>9.4f}")

    if unread:
        print("\nno recognisable metrics in:")
        for name, keys in unread:
            print(f"  {name}  -- top-level keys: {keys}")
        print("  add the real key name to PREFERRED or ALIASES at the top of "
              "this file.")

    print(f"\nwrote {a.results_dir / 'metrics.json'}  ({len(runs)} runs)")


if __name__ == "__main__":
    main()