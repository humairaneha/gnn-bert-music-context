"""
contrastive.py -- Task 4 contrastive dual encoder.

Importable:
    DualEncoder   two towers, no fusion. GNN -> g, frozen BERT CLS -> t, both
                  projected to a shared space and L2-normalised.
    info_nce      symmetric InfoNCE over in-batch negatives, with same-caption
                  pairs masked out of the negatives.

Runnable:
    python src/contrastive.py        # trains on MusicCaps

Delegates to run_musiccaps_task4.py, which points task4_contrastive.py at the
MusicCaps graphs and its own BERT cache.
"""
import runpy
from pathlib import Path

from task4_contrastive import DualEncoder, info_nce, retrieval_metrics

__all__ = ["DualEncoder", "info_nce", "retrieval_metrics"]

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_musiccaps_task4.py")),
                   run_name="__main__")
