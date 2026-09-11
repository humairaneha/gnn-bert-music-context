"""Train the MagnaTagATune audio-only GNN baseline.

Loads cached graphs from MTAT_graphs.py and predicts genre and mood tags.
Metrics include pooled and category-specific F1 and AUC-PR. Decision thresholds
are selected on validation data and then applied to test data; fixed-0.5 scores
are saved separately. Training produces a checkpoint, thresholds, metrics,
learning curves, and genre/mood embedding visualizations.

Fusion and no-graph models import shared training and evaluation utilities
from this module. Set dataset and model constants in the configuration block.

    python src/train.py --task 3
"""

from __future__ import annotations

import csv
import json
from collections import Counter
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool


# ================= CONFIG =================

DATA_DIR = Path("data/processed/mtat/graphs")
LABEL_SPACE = Path("data/processed/mtat/label_space.json")
RESULT_DIR = Path("results/task3_gnn_only")
FEATURE = "mfcc"                 # must match what mtat_graphs.py built
EDGE_POLICY = "tau"              # which graph build to train on: "tau" | "topk" | "percentile"
RUN_NAME = f"{FEATURE}_{EDGE_POLICY}"

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-6
HIDDEN = 128
NUM_LAYERS = 3
DROPOUT = 0.3
PATIENCE = 15                    # was 5 -- too tight for a 100-epoch budget

POS_WEIGHT_MAX = 10.0            # cap on (neg/pos); uncapped gives rare tags ~300
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)
TSNE_MAX_POINTS = 3000           # t-SNE is O(n^2); subsample above this
TSNE_TOP_CLASSES = 10            # colour the k most common tags, rest = "other"
ZERO_EDGE_WARN = 0.30            # flag if this share of clips has no similarity edge

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ================= MODEL =================

class GraphSAGE(nn.Module):
    """
    Three SAGEConv layers then mean pooling. With 5-segment clips, three layers
    already cover the whole graph, so depth beyond this only adds parameters.

    Returns the pooled graph embedding g on request -- that is the vector the
    t-SNE plots and the fusion models consume.
    """

    def __init__(self, in_dim, hidden, out_dim, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, out_dim)

    def forward(self, data, return_embedding=False):
        h = data.x
        for i, conv in enumerate(self.convs):
            h = F.relu(conv(h, data.edge_index))
            if i < len(self.convs) - 1:
                h = self.dropout(h)
        g = global_mean_pool(h, data.batch)
        logits = self.classifier(g)
        return (logits, g, h) if return_embedding else logits


# ================= METRICS =================

def _targets(batch):
    y = batch.y.float()
    return y.squeeze(1) if y.dim() == 3 else y


@torch.no_grad()
def predict(model, loader):
    """Returns (y_true, probs) as numpy arrays -- no thresholding here."""
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        ps.append(torch.sigmoid(model(batch)).cpu())
        ys.append(_targets(batch).cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def safe_auc_pr(y, p):
    """
    Mean average-precision over labels that actually have positives.

    average_precision_score(average="macro") does not skip empty labels; it
    returns nan for them and drags the mean to nan, or worse, silently reports
    a value computed over a different label set than you think.
    """
    scores = [average_precision_score(y[:, k], p[:, k])
              for k in range(y.shape[1]) if y[:, k].sum() > 0]
    return float(np.mean(scores)) if scores else float("nan")


def score(y, p, thresholds=None, n_genre=None):
    """Full metric set at the given thresholds (0.5 if none)."""
    t = 0.5 if thresholds is None else np.asarray(thresholds)
    pred = (p >= t).astype(int)

    out = {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "auc_pr": safe_auc_pr(y, p),
    }
    if n_genre is not None:
        for name, sl in (("genre", slice(0, n_genre)), ("mood", slice(n_genre, None))):
            out[f"{name}_macro_f1"] = float(
                f1_score(y[:, sl], pred[:, sl], average="macro", zero_division=0))
            out[f"{name}_auc_pr"] = safe_auc_pr(y[:, sl], p[:, sl])
    return out, pred


def find_best_thresholds(probs, labels, grid=THRESHOLD_GRID):
    """
    One threshold per label, maximising F1 on the VALIDATION split.

    This must never see test. Tuning K thresholds on test would be fitting K
    free parameters to the evaluation set, and with rare tags the gain is large
    enough that it would badly flatter the result.
    """
    thresholds = np.full(probs.shape[1], 0.5)
    for k in range(probs.shape[1]):
        if labels[:, k].sum() == 0:
            continue                       # nothing to tune against
        scores = [f1_score(labels[:, k], (probs[:, k] >= t).astype(int),
                           zero_division=0) for t in grid]
        thresholds[k] = grid[int(np.argmax(scores))]
    return thresholds


def make_tag_weights(train_graphs, cap=POS_WEIGHT_MAX):
    """
    Inverse-frequency pos_weight for BCEWithLogitsLoss, capped.

    The cap matters. Uncapped, (neg/pos) for a tag with ~57 positives in 12k
    clips is over 200, which pushes that logit up so hard the model fires on
    everything -- recall near 1, precision near 0. Capping trades a little
    rare-tag recall for a usable precision/recall balance, and the per-tag
    thresholds pick up the rest.
    """
    labels = torch.cat([g.y for g in train_graphs], dim=0)
    positives = labels.sum(dim=0)
    negatives = len(labels) - positives
    return torch.clamp(negatives / (positives + 1e-6), min=1.0, max=cap).float()


# ================= REPORTING =================

def save_per_label_metrics(y, pred, p, labels, thresholds, groups, path):
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, average=None, zero_division=0)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "group", "threshold", "support",
                    "precision", "recall", "f1", "auc_pr"])
        for i, name in enumerate(labels):
            ap = (average_precision_score(y[:, i], p[:, i])
                  if y[:, i].sum() > 0 else float("nan"))
            w.writerow([name, groups[i], round(float(thresholds[i]), 3),
                        int(support[i]), round(float(precision[i]), 4),
                        round(float(recall[i]), 4), round(float(f1[i]), 4),
                        round(float(ap), 4)])


def plot_curves(history, path):
    """Twin axis -- loss and F1 live on different scales."""
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(history["train_loss"], label="train loss", color="#1f77b4")
    ax1.plot(history["val_loss"], label="val loss", color="#1f77b4", ls="--")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("BCE loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(history["val_macro_f1"], label="val Macro-F1", color="#d62728")
    ax2.plot(history["auc_pr"], label="val AUC-PR", color="#2ca02c")
    ax2.set_ylabel("F1 / AUC-PR", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    ax1.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ================= COLOURED t-SNE =================

def primary_label(y_block, names, train_counts):
    """
    Multi-label data has no single colour, so pick one: the RAREST positive tag.

    Rarest rather than most-frequent because it is the most specific thing said
    about the clip. Colouring by the most frequent tag would paint half the plot
    "classical" and hide exactly the structure the plot is for. Clips with no
    positive tag in this group return None and are dropped from that panel.
    """
    order = np.argsort(train_counts)          # rarest first
    out = []
    for row in y_block:
        pos = [i for i in order if row[i] > 0]
        out.append(names[pos[0]] if pos else None)
    return out


def plot_tsne(embeddings, y, labels, n_genre, train_counts, out_dir,
              max_points=TSNE_MAX_POINTS, top_k=TSNE_TOP_CLASSES, seed=SEED):
    """
    Two panels of the SAME embedding: coloured by genre, coloured by mood.
    Plot graph embeddings in separate genre and mood panels.

    Only the top_k most common primary labels get their own colour; everything
    else is grey "other", because 35 genres in one legend is unreadable and the
    tail classes have too few points to form visible structure anyway.
    """
    rng = np.random.default_rng(seed)
    if len(embeddings) > max_points:
        idx = rng.choice(len(embeddings), max_points, replace=False)
        embeddings, y = embeddings[idx], y[idx]

    coords = TSNE(n_components=2, random_state=seed,
                  perplexity=min(30, max(5, len(embeddings) // 100)),
                  init="pca").fit_transform(embeddings)

    panels = [
        ("genre", y[:, :n_genre], labels[:n_genre], train_counts[:n_genre]),
        ("mood", y[:, n_genre:], labels[n_genre:], train_counts[n_genre:]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, (title, block, names, counts) in zip(axes, panels):
        prim = primary_label(block, names, counts)
        present = [c for c in prim if c is not None]
        keep = [c for c, _ in Counter(present).most_common(top_k)]
        palette = plt.cm.tab20(np.linspace(0, 1, len(keep)))

        other = [i for i, c in enumerate(prim) if c is not None and c not in keep]
        if other:
            ax.scatter(coords[other, 0], coords[other, 1], s=6, c="#d9d9d9",
                       label=f"other ({len(other)})", zorder=1)
        for colour, cls in zip(palette, keep):
            sel = [i for i, c in enumerate(prim) if c == cls]
            ax.scatter(coords[sel, 0], coords[sel, 1], s=10, color=colour,
                       label=f"{cls} ({len(sel)})", zorder=2)

        n_none = sum(c is None for c in prim)
        ax.set_title(f"GNN graph embedding, coloured by {title}"
                     + (f"  ({n_none} clips have no {title} tag, not shown)"
                        if n_none else ""))
        ax.legend(fontsize=7, markerscale=1.6, loc="best", framealpha=.9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    path = Path(out_dir) / "tsne_genre_mood.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


@torch.no_grad()
def export_embeddings(model, loader, path):
    """Pooled graph embeddings g, kept for the fusion models and the t-SNE."""
    model.eval()
    gs, ys, tids = [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        _, g, _ = model(batch, return_embedding=True)
        gs.append(g.cpu())
        ys.append(_targets(batch).cpu())
        tids.extend(list(batch.track_id))
    g = torch.cat(gs)
    y = torch.cat(ys)
    torch.save({"g": g, "y": y, "track_id": tids}, path)
    return g.numpy(), y.numpy()


# ================= MAIN =================

def main():
    out_dir = RESULT_DIR / RUN_NAME        # one folder per ablation row
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE} | feature: {FEATURE} | edges: {EDGE_POLICY}\n")

    def graphs_for(split):
        path = DATA_DIR / f"{split}_{FEATURE}_{EDGE_POLICY}.pt"
        if not path.exists():                # pre-ablation filename
            path = DATA_DIR / f"{split}_{FEATURE}.pt"
        return torch.load(path, weights_only=False)

    train, val, test = graphs_for("train"), graphs_for("val"), graphs_for("test")
    print(f"graphs: train={len(train)}, val={len(val)}, test={len(test)}")

    space = json.loads(LABEL_SPACE.read_text())
    labels = space["genre"] + space["mood"]
    n_genre = space["n_genre"]
    groups = ["genre"] * n_genre + ["mood"] * space["n_mood"]
    print(f"targets: {n_genre} genre + {space['n_mood']} mood = {len(labels)}")

    # sanity: a silently mismatched label space makes every per-tag number wrong
    assert train[0].y.shape[-1] == len(labels), (
        f"graphs have {train[0].y.shape[-1]} targets but label_space.json lists "
        f"{len(labels)} -- rebuild the graphs after changing the tag partition")

    train_counts = torch.cat([g.y for g in train]).sum(0).numpy()
    n_sim = np.array([int((g.edge_attr[:, 0] < 0.5).sum()) // 2 for g in train])
    print(f"similarity edges per clip: mean {n_sim.mean():.2f}, "
          f"median {np.median(n_sim):.0f}, "
          f"bare temporal chain {(n_sim == 0).mean():.1%} of clips")
    if (n_sim == 0).mean() > ZERO_EDGE_WARN:
        print(f"  ^ above {ZERO_EDGE_WARN:.0%}: for that share of clips the GNN sees "
              f"only time order.\n    Set EDGE_POLICY = 'topk' in mtat_graphs.py and "
              f"rebuild before trusting a\n    'graph structure helps' claim. "
              f"(GTZAN at tau=0.3 gave 1.84 edges/track.)\n")

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    model = GraphSAGE(train[0].x.shape[1], HIDDEN, len(labels)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=make_tag_weights(train).to(DEVICE))

    best, wait = -1.0, 0
    history = {k: [] for k in ("train_loss", "val_loss", "val_macro_f1",
                               "micro_f1", "auc_pr")}
    ckpt = out_dir / "best_gnn.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch), _targets(batch))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        y_val, p_val = predict(model, val_loader)
        val_metrics, _ = score(y_val, p_val, n_genre=n_genre)
        with torch.no_grad():
            val_loss = float(np.mean([
                criterion(model(b.to(DEVICE)), _targets(b.to(DEVICE))).item()
                for b in val_loader]))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["micro_f1"].append(val_metrics["micro_f1"])
        history["auc_pr"].append(val_metrics["auc_pr"])

        # selection on AUC-PR, not thresholded F1: AUC-PR measures ranking, and
        # the thresholds are tuned after training anyway
        marker = ""
        if val_metrics["auc_pr"] > best:
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

    # ---- evaluate ----
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    y_val, p_val = predict(model, val_loader)
    thresholds = find_best_thresholds(p_val, y_val)     # VAL only
    np.save(out_dir / "thresholds.npy", thresholds)

    y_test, p_test = predict(model, test_loader)
    at_half, _ = score(y_test, p_test, None, n_genre)
    tuned, pred = score(y_test, p_test, thresholds, n_genre)

    rng = np.random.default_rng(SEED)
    prior = y_val.mean(axis=0)
    p_rand = rng.random(y_test.shape) * 0 + prior            # predict the prior
    baseline, _ = score(y_test, p_rand, prior, n_genre)

    results = {"run": RUN_NAME, "feature": FEATURE, "edge_policy": EDGE_POLICY,
               "zero_edge_fraction_train": round(float((n_sim == 0).mean()), 4),
               "similarity_edges_mean_train": round(float(n_sim.mean()), 2),
               "test_at_0.5": at_half, "test_tuned": tuned,
               "baseline_prior": baseline,
               "n_genre": n_genre, "n_mood": space["n_mood"]}
    json.dump(results, open(out_dir / "test_metrics.json", "w"), indent=2)

    print("\n" + "=" * 72)
    print(f"{'':<22}{'Macro-F1':>10}{'Micro-F1':>10}{'AUC-PR':>10}"
          f"{'genre F1':>11}{'mood F1':>10}")
    for name, m in (("baseline (prior)", baseline), ("test @ 0.5", at_half),
                    ("test @ tuned", tuned)):
        print(f"{name:<22}{m['macro_f1']:>10.4f}{m['micro_f1']:>10.4f}"
              f"{m['auc_pr']:>10.4f}{m['genre_macro_f1']:>11.4f}"
              f"{m['mood_macro_f1']:>10.4f}")
    print("=" * 72)
    print(f"threshold tuning: Macro-F1 {at_half['macro_f1']:.4f} -> "
          f"{tuned['macro_f1']:.4f}  ({tuned['macro_f1'] - at_half['macro_f1']:+.4f})")

    save_per_label_metrics(y_test, pred, p_test, labels, thresholds, groups,
                           out_dir / "per_label_metrics.csv")
    np.save(out_dir / "label_confusion_matrices.npy",
            multilabel_confusion_matrix(y_test, pred))

    g_test, y_emb = export_embeddings(model, test_loader,
                                      out_dir / "gnn_embeddings.pt")
    path = plot_tsne(g_test, y_emb, labels, n_genre, train_counts, out_dir)
    print(f"\nwrote {out_dir}/  (metrics, per-label CSV, curves, {path.name})")
    return results


if __name__ == "__main__":
    main()