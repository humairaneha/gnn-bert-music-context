"""
run_musiccaps_supervised.py -- supervised GNN on the MusicCaps graphs.

The Task 4 deliverable asks for zero-shot tag prediction "vs the Task 3
supervised model". Your Task 3 model was trained on MagnaTagATune, with a
different label space, different audio and different splits -- so comparing a
MusicCaps zero-shot number against it would be meaningless. This produces the
supervised reference on the SAME data, which is what makes that comparison real.

It reuses task3_gnn_only.py unchanged and only overrides the paths, so the model,
loss, pos_weight cap, threshold tuning and metrics are identical to every other
row in the project.

    python src/run_musiccaps_supervised.py
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
