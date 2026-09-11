"""Train a no-graph audio baseline for MagnaTagATune.

Reads the cached segment graphs but does not use their edges. Segment vectors
are mean-pooled and passed through an MLP. Splits, loss utilities, threshold
tuning, metrics, and early-stopping settings are shared with gnn.py. The model
provides a comparison for the GNN's use of graph structure.

Build the graph caches with src/graph_builder.py --dataset mtat first.

    python src/train.py --task 3 --variant mlp
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import multilabel_confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Every shared piece is imported, never copied -- a divergence between the two
# scripts' metric code would silently invalidate the comparison.
from gnn import (
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, HIDDEN, LABEL_SPACE, LR, NUM_LAYERS,
    PATIENCE, SEED, TSNE_MAX_POINTS, WEIGHT_DECAY,
    find_best_thresholds, make_tag_weights, plot_curves, plot_tsne,
    safe_auc_pr, save_per_label_metrics, score,
)


# ================= CONFIG =================

DATA_DIR = Path("data/processed/mtat/graphs")
RESULT_DIR = Path("results/task3_mlp_nograph")
FEATURE = "mfcc"
EDGE_POLICY = "tau"          # only picks WHICH cached file to read; edges are discarded
RUN_NAME = f"{FEATURE}_nograph"

# "mean"     -- the strict control. global_mean_pool is exactly what the GNN does
#               to its node embeddings, so this gives the MLP the same summary
#               the GNN ends up with, minus message passing.
# "mean+std" -- adds cross-segment variance the GNN could in principle compute.
# "flatten"  -- all 5 segments in order (130-d). Gives the MLP segment ORDER,
#               which the GNN's mean pooling actually discards, so this can beat
#               the GNN for a reason unrelated to graph structure. Informative,
#               but do not report it as "the no-graph baseline".
POOLING = "mean"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================= DATA =================

def graphs_for(split):
    path = DATA_DIR / f"{split}_{FEATURE}_{EDGE_POLICY}.pt"
    if not path.exists():
        path = DATA_DIR / f"{split}_{FEATURE}.pt"
    return torch.load(path, weights_only=False)


def collapse(graphs, pooling=POOLING):
    """
    One vector per clip, edges discarded. Returns (X, Y) float tensors.

    `edge_index` is never touched -- that is the entire point of this file.
    """
    xs, ys = [], []
    for g in graphs:
        if pooling == "mean":
            v = g.x.mean(dim=0)
        elif pooling == "mean+std":
            v = torch.cat([g.x.mean(dim=0), g.x.std(dim=0, unbiased=False)])
        elif pooling == "flatten":
            v = g.x.reshape(-1)
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
        xs.append(v)
        ys.append(g.y.squeeze(0))
    return torch.stack(xs).float(), torch.stack(ys).float()


# ================= MODEL =================

class MLP(nn.Module):
    """
    Depth and width matched to the GNN (NUM_LAYERS hidden layers at HIDDEN
    units, same dropout), so the two models have comparable capacity and the
    only structural difference is the absence of message passing.
    """

    def __init__(self, in_dim, hidden, out_dim, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden, out_dim)

    def forward(self, x, return_embedding=False):
        h = self.body(x)
        logits = self.classifier(h)
        return (logits, h) if return_embedding else logits


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        ps.append(torch.sigmoid(model(xb.to(DEVICE))).cpu())
        ys.append(yb)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


@torch.no_grad()
def embeddings(model, loader):
    model.eval()
    hs = []
    for xb, _ in loader:
        _, h = model(xb.to(DEVICE), return_embedding=True)
        hs.append(h.cpu())
    return torch.cat(hs).numpy()


# ================= MAIN =================

def main():
    out_dir = RESULT_DIR / RUN_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE} | feature: {FEATURE} | pooling: {POOLING} | NO GRAPH\n")

    train_g, val_g, test_g = graphs_for("train"), graphs_for("val"), graphs_for("test")
    X_tr, Y_tr = collapse(train_g)
    X_va, Y_va = collapse(val_g)
    X_te, Y_te = collapse(test_g)
    print(f"clips: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")
    print(f"input dim: {X_tr.shape[1]} (GNN saw {train_g[0].x.shape[0]} nodes x "
          f"{train_g[0].x.shape[1]}-d + edges)")

    space = json.loads(LABEL_SPACE.read_text())
    labels = space["genre"] + space["mood"]
    n_genre = space["n_genre"]
    groups = ["genre"] * n_genre + ["mood"] * space["n_mood"]
    assert Y_tr.shape[1] == len(labels), "label space does not match the graphs"
    print(f"targets: {n_genre} genre + {space['n_mood']} mood = {len(labels)}\n")

    train_counts = Y_tr.sum(0).numpy()
    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, Y_va), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(X_te, Y_te), batch_size=BATCH_SIZE)

    model = MLP(X_tr.shape[1], HIDDEN, len(labels)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # same capped inverse-frequency weights the GNN used
    criterion = nn.BCEWithLogitsLoss(pos_weight=make_tag_weights(train_g).to(DEVICE))

    best, wait = -1.0, 0
    history = {k: [] for k in ("train_loss", "val_loss", "val_macro_f1",
                               "micro_f1", "auc_pr")}
    ckpt = out_dir / "best_mlp.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        y_val, p_val = predict(model, val_loader)
        val_metrics, _ = score(y_val, p_val, n_genre=n_genre)
        with torch.no_grad():
            val_loss = float(np.mean([
                criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
                for xb, yb in val_loader]))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["micro_f1"].append(val_metrics["micro_f1"])
        history["auc_pr"].append(val_metrics["auc_pr"])

        marker = ""
        if val_metrics["auc_pr"] > best:          # same selection rule as the GNN
            best, wait = val_metrics["auc_pr"], 0
            torch.save(model.state_dict(), ckpt)
            marker = "  <- saved"
        else:
            wait += 1

        print(f"epoch {epoch:3d} | loss {train_loss:.4f}/{val_loss:.4f} | "
              f"val macroF1 {val_metrics['macro_f1']:.4f} | "
              f"AUC-PR {val_metrics['auc_pr']:.4f}{marker}")
        if wait >= PATIENCE:
            print(f"early stopped at epoch {epoch}")
            break

    json.dump(history, open(out_dir / "training_history.json", "w"), indent=2)
    plot_curves(history, out_dir / "loss_f1_curves.png")

    # ---- evaluate, identical protocol to the GNN ----
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    y_val, p_val = predict(model, val_loader)
    thresholds = find_best_thresholds(p_val, y_val)      # VAL only
    np.save(out_dir / "thresholds.npy", thresholds)

    y_test, p_test = predict(model, test_loader)
    at_half, _ = score(y_test, p_test, None, n_genre)
    tuned, pred = score(y_test, p_test, thresholds, n_genre)

    results = {"run": RUN_NAME, "model": "MLP (no graph)", "pooling": POOLING,
               "input_dim": int(X_tr.shape[1]),
               "test_at_0.5": at_half, "test_tuned": tuned,
               "n_genre": n_genre, "n_mood": space["n_mood"]}
    json.dump(results, open(out_dir / "test_metrics.json", "w"), indent=2)

    print("\n" + "=" * 72)
    print(f"{'':<22}{'Macro-F1':>10}{'Micro-F1':>10}{'AUC-PR':>10}"
          f"{'genre F1':>11}{'mood F1':>10}")
    for name, m in (("MLP @ 0.5", at_half), ("MLP @ tuned", tuned)):
        print(f"{name:<22}{m['macro_f1']:>10.4f}{m['micro_f1']:>10.4f}"
              f"{m['auc_pr']:>10.4f}{m['genre_macro_f1']:>11.4f}"
              f"{m['mood_macro_f1']:>10.4f}")
    print("=" * 72)

    gnn = Path("results/task3_gnn_only") / f"{FEATURE}_{EDGE_POLICY}" / "test_metrics.json"
    if gnn.exists():
        g = json.loads(gnn.read_text())["test_tuned"]
        d_f1 = tuned["macro_f1"] - g["macro_f1"]
        d_ap = tuned["auc_pr"] - g["auc_pr"]
        print(f"\nvs GNN ({FEATURE}_{EDGE_POLICY}): "
              f"Macro-F1 {g['macro_f1']:.4f} -> {tuned['macro_f1']:.4f} ({d_f1:+.4f}), "
              f"AUC-PR {g['auc_pr']:.4f} -> {tuned['auc_pr']:.4f} ({d_ap:+.4f})")
        print("  the graph is earning its place." if d_ap < -0.005 else
              "  the graph is NOT beating a plain MLP on the same features. "
              "Report this\n  honestly -- it is a real finding about 5-node segment "
              "graphs, not a bug.")
    else:
        print(f"\n(run task3_gnn_only.py first to get the side-by-side comparison)")

    save_per_label_metrics(y_test, pred, p_test, labels, thresholds, groups,
                           out_dir / "per_label_metrics.csv")
    np.save(out_dir / "label_confusion_matrices.npy",
            multilabel_confusion_matrix(y_test, pred))

    h_test = embeddings(model, test_loader)
    torch.save({"h": torch.from_numpy(h_test), "y": torch.from_numpy(y_test)},
               out_dir / "mlp_embeddings.pt")
    path = plot_tsne(h_test, y_test, labels, n_genre, train_counts, out_dir)
    print(f"\nwrote {out_dir}/  (metrics, per-label CSV, curves, {path.name})")
    return results


if __name__ == "__main__":
    main()
