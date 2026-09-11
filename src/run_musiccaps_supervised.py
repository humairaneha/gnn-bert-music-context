"""Train the supervised audio-only GNN reference on MusicCaps.

Reuses gnn.py with MusicCaps paths and label metadata. This provides a
supervised tagging reference on the retrieval dataset, avoiding comparison
with the different MagnaTagATune label space. It is an audio-only reference,
not a supervised GNN–BERT fusion model.

    python src/train.py --task 4 --variant supervised
"""

from pathlib import Path

import gnn as t3

# --- point the module at MusicCaps -------------------------------------------
t3.DATA_DIR = Path("data/processed/musiccaps/graphs")
t3.LABEL_SPACE = Path("data/processed/musiccaps/label_space.json")
t3.RESULT_DIR = Path("results/musiccaps_supervised")
t3.FEATURE = "mfcc"
t3.EDGE_POLICY = "tau"                 # match what musiccaps_graphs.py built
t3.RUN_NAME = f"{t3.FEATURE}_{t3.EDGE_POLICY}"   # recomputed: it was set at import

# MusicCaps train is ~3.7k clips against MTAT's 12.2k, so epochs are cheap.
# Keep PATIENCE as-is; selection is on validation AUC-PR either way.
t3.EPOCHS = 60

if __name__ == "__main__":
    print(f"supervised GNN on MusicCaps -> {t3.RESULT_DIR / t3.RUN_NAME}\n")
    t3.main()
