"""Train and evaluate a genre classifier on cached GTZAN MFCC graphs.

Inputs:
    data/processed/GTZAN/graphs/{train,val,test}_mfcc_<EDGE_POLICY>.pt
    data/processed/GTZAN/label_space.json

Build the graphs with src/GTZAN_graphs.py first. This module loads the graph
features, trains with cross-entropy, selects a validation checkpoint, and saves
predictions, metrics, curves, confusion matrices, and embeddings under
results/task2/<RUN_NAME>/.

The CNN comparison uses a stored segment-level validation reference; the GNN
scores are at track level. These evaluation units differ.

    python src/train.py --task 2
"""

from __future__ import annotations

import csv
import json
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
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    global_mean_pool,
)


# ===========================================================================
# 0. CONFIGURATION -- edit only this block
# ===========================================================================

DATA_DIR = Path("data/processed/GTZAN")
GRAPH_DIR = DATA_DIR / "graphs"
LABEL_SPACE = DATA_DIR / "label_space.json"

RESULT_DIR = Path("results/task2")

FEATURE = "mfcc"

# Match the policy used by GTZAN_graphs.py:
#   "tau" | "percentile" | "topk"
EDGE_POLICY = "tau"

# MFCC-only experiment backbone.
#   "sage" | "gat"
# GraphSAGE is the default used for this experiment.
BACKBONE = "sage"

RUN_NAME = f"{FEATURE}_{BACKBONE}_{EDGE_POLICY}"

EPOCHS = 100
BATCH_SIZE = 32

LR = 1e-3
WEIGHT_DECAY = 1e-5

HIDDEN_DIM = 64
NUM_LAYERS = 3

GAT_HEADS = 4
GAT_DROPOUT = 0.3
SAGE_DROPOUT = 0.0

PATIENCE = 15

# Checkpoint selection constraint:
# a new validation best is rejected when train-val Macro-F1 gap is too large.
MAX_OVERFIT_GAP = 0.20

TSNE_MAX_POINTS = 3000

# ---------------------------------------------------------------------------
# CNN comparison reference
# ---------------------------------------------------------------------------
# Stored CNN reference: segment-level validation Macro-F1 0.7470 and
# Micro-F1 0.7508. Its source notebook is not included in this checkout.
# The GNN is evaluated per track, so these scores use different units.
CNN_REFERENCE_NAME = "CNN mel-spectrogram"
CNN_REFERENCE_EVAL_LEVEL = "segment-level"
CNN_REFERENCE_SPLIT = "validation"
CNN_VAL_MACRO_F1 = 0.7470
CNN_VAL_MICRO_F1 = 0.7508

SEED = 42
VERBOSE = True


# ===========================================================================
# 1. Reproducibility + device
# ===========================================================================

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


# ===========================================================================
# 2. Load saved graphs + label space
# ===========================================================================

def graph_path(split: str) -> Path:
    """
    Graph filename created by mfcc_graphs.py.
    """
    return (
        GRAPH_DIR
        / f"{split}_{FEATURE}_{EDGE_POLICY}.pt"
    )


def load_graphs(split: str):
    path = graph_path(split)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing graph file: {path}\n"
            "Run mfcc_graphs.py first using the same EDGE_POLICY."
        )

    return torch.load(
        path,
        weights_only=False,
    )


def load_label_space():
    if not LABEL_SPACE.exists():
        raise FileNotFoundError(
            f"Missing label space: {LABEL_SPACE}"
        )

    space = json.loads(
        LABEL_SPACE.read_text(
            encoding="utf-8"
        )
    )

    labels = list(
        space.get("genre", [])
    )

    if not labels:
        labels = list(
            space.get("tags", [])
        )

    if not labels:
        raise ValueError(
            "label_space.json must contain `genre` "
            "or a usable `tags` list."
        )

    return space, labels


def validate_graphs(
    train,
    val,
    test,
    labels,
):
    """
    Ensure saved graphs match the label space and are valid
    for single-label CrossEntropy training.
    """
    all_graphs = train + val + test

    if not all_graphs:
        raise ValueError(
            "No graphs were loaded."
        )

    for split_name, graphs in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        if not graphs:
            raise ValueError(
                f"{split_name} graph list is empty."
            )

        for i, graph in enumerate(graphs):
            if graph.y.numel() != 1:
                raise ValueError(
                    f"{split_name}[{i}] has y shape "
                    f"{tuple(graph.y.shape)}. "
                    "This MFCC training file expects one "
                    "integer class target per graph."
                )

    targets = torch.cat([
        graph.y.reshape(-1)
        for graph in all_graphs
    ])

    min_target = int(
        targets.min().item()
    )

    max_target = int(
        targets.max().item()
    )

    if min_target < 0:
        raise ValueError(
            f"Negative class index found: {min_target}"
        )

    if max_target >= len(labels):
        raise ValueError(
            f"Graph class index reaches {max_target}, "
            f"but label_space contains only "
            f"{len(labels)} classes."
        )

    in_dim = int(
        train[0].x.shape[1]
    )

    for split_name, graphs in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        bad = [
            i
            for i, graph in enumerate(graphs)
            if int(graph.x.shape[1]) != in_dim
        ]

        if bad:
            raise ValueError(
                f"{split_name} contains graphs with "
                f"inconsistent node feature dimensions. "
                f"Examples: {bad[:10]}"
            )

    return in_dim


# ===========================================================================
# 3. GNN model -- same MFCC-only architecture family as mfcc_only.py
# ===========================================================================

EDGE_ATTR_DIM = 2


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=SAGE_DROPOUT,
    ):
        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()

        self.convs.append(
            SAGEConv(
                in_dim,
                hidden_dim,
            )
        )

        for _ in range(
            num_layers - 1
        ):
            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    hidden_dim,
                )
            )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):
        # SAGEConv does not use edge_attr.
        for conv in self.convs:
            x = F.relu(
                conv(
                    x,
                    edge_index,
                )
            )

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

        return global_mean_pool(
            x,
            batch,
        )


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        heads=GAT_HEADS,
        edge_dim=EDGE_ATTR_DIM,
        dropout=GAT_DROPOUT,
    ):
        super().__init__()

        self.dropout = dropout

        self.convs = nn.ModuleList()

        self.convs.append(
            GATConv(
                in_dim,
                hidden_dim,
                heads=heads,
                concat=False,
                edge_dim=edge_dim,
            )
        )

        for _ in range(
            num_layers - 1
        ):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=edge_dim,
                )
            )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
    ):
        for conv in self.convs:
            x = F.relu(
                conv(
                    x,
                    edge_index,
                    edge_attr=edge_attr,
                )
            )

            x = F.dropout(
                x,
                p=self.dropout,
                training=self.training,
            )

        return global_mean_pool(
            x,
            batch,
        )


def make_encoder(
    backbone,
    in_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
):
    if backbone == "sage":
        return GraphSAGEEncoder(
            in_dim,
            hidden_dim,
            num_layers,
        )

    if backbone == "gat":
        return GATEncoder(
            in_dim,
            hidden_dim,
            num_layers,
        )

    raise ValueError(
        f"Unknown BACKBONE={backbone!r}. "
        "Use 'sage' or 'gat'."
    )


class SingleBranchGNN(nn.Module):
    """
    MFCC graph encoder + graph-level genre classifier.
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_layers=NUM_LAYERS,
        backbone=BACKBONE,
    ):
        super().__init__()

        self.encoder = make_encoder(
            backbone,
            in_dim,
            hidden_dim,
            num_layers,
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes,
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        return_embedding=False,
    ):
        g = self.encoder(
            x,
            edge_index,
            edge_attr,
            batch,
        )

        logits = self.classifier(
            g
        )

        if return_embedding:
            return logits, g

        return logits


# ===========================================================================
# 4. Prediction + metrics
# ===========================================================================

@torch.no_grad()
def predict(
    model,
    loader,
):
    """
    Returns:
        y_true : [N]
        y_pred : [N]
        probs  : [N, C]
    """
    model.eval()

    all_y = []
    all_pred = []
    all_probs = []

    for batch in loader:
        batch = batch.to(
            DEVICE
        )

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        probs = torch.softmax(
            logits,
            dim=1,
        )

        pred = probs.argmax(
            dim=1
        )

        all_y.append(
            batch.y.reshape(-1).cpu()
        )

        all_pred.append(
            pred.cpu()
        )

        all_probs.append(
            probs.cpu()
        )

    return (
        torch.cat(all_y).numpy(),
        torch.cat(all_pred).numpy(),
        torch.cat(all_probs).numpy(),
    )


def multiclass_auc_pr(
    y_true,
    probs,
    num_classes,
):
    """
    One-vs-rest macro Average Precision.

    This is the multiclass analogue of the AUC-PR metric used in
    the multi-label GNN file.
    """
    y_bin = label_binarize(
        y_true,
        classes=np.arange(
            num_classes
        ),
    )

    # Binary edge case.
    if num_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.column_stack([
            1 - y_bin[:, 0],
            y_bin[:, 0],
        ])

    scores = []

    for class_idx in range(
        num_classes
    ):
        if y_bin[:, class_idx].sum() == 0:
            continue

        scores.append(
            average_precision_score(
                y_bin[:, class_idx],
                probs[:, class_idx],
            )
        )

    return (
        float(np.mean(scores))
        if scores
        else float("nan")
    )


def score(
    y_true,
    y_pred,
    probs,
    num_classes,
):
    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "micro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="micro",
                    zero_division=0,
                )
            ),

        "auc_pr":
            multiclass_auc_pr(
                y_true,
                probs,
                num_classes,
            ),
    }


def evaluate(
    model,
    loader,
    num_classes,
):
    y_true, y_pred, probs = predict(
        model,
        loader,
    )

    metrics = score(
        y_true,
        y_pred,
        probs,
        num_classes,
    )

    return (
        metrics,
        y_true,
        y_pred,
        probs,
    )


# ===========================================================================
# 5. Loss evaluation
# ===========================================================================

@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    criterion,
):
    model.eval()

    losses = []
    sizes = []

    for batch in loader:
        batch = batch.to(
            DEVICE
        )

        logits = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        y = batch.y.reshape(-1)

        loss = criterion(
            logits,
            y,
        )

        losses.append(
            float(
                loss.item()
            )
        )

        sizes.append(
            int(
                batch.num_graphs
            )
        )

    if not sizes:
        return float("nan")

    return float(
        np.average(
            losses,
            weights=sizes,
        )
    )


# ===========================================================================
# 6. Evaluation-file writers
# ===========================================================================

def save_per_class_metrics(
    y_true,
    y_pred,
    labels,
    path,
):
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=np.arange(
                len(labels)
            ),
            zero_division=0,
        )
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "class_index",
            "label",
            "support",
            "precision",
            "recall",
            "f1",
        ])

        for i, label in enumerate(
            labels
        ):
            writer.writerow([
                i,
                label,
                int(support[i]),
                round(
                    float(precision[i]),
                    6,
                ),
                round(
                    float(recall[i]),
                    6,
                ),
                round(
                    float(f1[i]),
                    6,
                ),
            ])


def save_classification_report(
    y_true,
    y_pred,
    labels,
    out_dir,
):
    label_indices = list(
        range(
            len(labels)
        )
    )

    label_names = [
        str(label)
        for label in labels
    ]

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=label_names,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=label_indices,
        target_names=label_names,
        zero_division=0,
    )

    (
        out_dir
        / "classification_report.json"
    ).write_text(
        json.dumps(
            report_dict,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        out_dir
        / "classification_report.txt"
    ).write_text(
        report_text,
        encoding="utf-8",
    )


def save_predictions(
    y_true,
    y_pred,
    probs,
    track_ids,
    labels,
    path,
):
    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "track_id",
            "true_index",
            "true_label",
            "pred_index",
            "pred_label",
            "confidence",
            "correct",
        ])

        for i in range(
            len(y_true)
        ):
            true_idx = int(
                y_true[i]
            )

            pred_idx = int(
                y_pred[i]
            )

            writer.writerow([
                track_ids[i],
                true_idx,
                labels[true_idx],
                pred_idx,
                labels[pred_idx],
                round(
                    float(
                        probs[
                            i,
                            pred_idx,
                        ]
                    ),
                    6,
                ),
                int(
                    true_idx
                    == pred_idx
                ),
            ])


# ===========================================================================
# 7. Plots
# ===========================================================================

def plot_loss_curve(
    history,
    path,
):
    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        history["train_loss"],
        label="train loss",
    )

    ax.plot(
        history["val_loss"],
        label="val loss",
    )

    ax.set_xlabel(
        "epoch"
    )

    ax.set_ylabel(
        "cross-entropy loss"
    )

    ax.set_title(
        "Training and validation loss"
    )

    ax.legend()
    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(
        fig
    )


def plot_f1_curve(
    history,
    path,
):
    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    ax.plot(
        history["train_macro_f1"],
        label="train Macro-F1",
    )

    ax.plot(
        history["val_macro_f1"],
        label="val Macro-F1",
    )

    ax.plot(
        history["val_micro_f1"],
        label="val Micro-F1",
    )

    ax.set_xlabel(
        "epoch"
    )

    ax.set_ylabel(
        "F1"
    )

    ax.set_title(
        "Training and validation F1"
    )

    ax.legend()
    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(
        fig
    )


def plot_confusion_matrix(
    cm,
    labels,
    path,
):
    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ticks = np.arange(
        len(labels)
    )

    ax.set_xticks(
        ticks
    )

    ax.set_yticks(
        ticks
    )

    ax.set_xticklabels(
        [
            str(x)
            for x in labels
        ],
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        [
            str(x)
            for x in labels
        ]
    )

    ax.set_xlabel(
        "predicted"
    )

    ax.set_ylabel(
        "true"
    )

    ax.set_title(
        "Test confusion matrix"
    )

    # Write counts inside cells when class count is manageable.
    if len(labels) <= 20:
        threshold = (
            cm.max() / 2
            if cm.size
            else 0
        )

        for i in range(
            cm.shape[0]
        ):
            for j in range(
                cm.shape[1]
            ):
                ax.text(
                    j,
                    i,
                    str(
                        int(
                            cm[i, j]
                        )
                    ),
                    ha="center",
                    va="center",
                )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(
        fig
    )


# ===========================================================================
# 8. Embedding export + t-SNE
# ===========================================================================

@torch.no_grad()
def export_embeddings(
    model,
    loader,
    path,
):
    model.eval()

    embeddings = []
    ys = []
    track_ids = []

    for batch in loader:
        batch = batch.to(
            DEVICE
        )

        _, graph_embedding = model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            return_embedding=True,
        )

        embeddings.append(
            graph_embedding.cpu()
        )

        ys.append(
            batch.y.reshape(-1).cpu()
        )

        track_ids.extend([
            str(track_id)
            for track_id
            in batch.track_id
        ])

    g = torch.cat(
        embeddings,
        dim=0,
    )

    y = torch.cat(
        ys,
        dim=0,
    )

    torch.save(
        {
            "g": g,
            "y": y,
            "track_id": track_ids,
        },
        path,
    )

    return (
        g.numpy(),
        y.numpy(),
        track_ids,
    )


def plot_tsne(
    embeddings,
    y,
    labels,
    path,
    max_points=TSNE_MAX_POINTS,
    seed=SEED,
):
    """
    Colored t-SNE of pooled GNN graph embeddings.

    Each genre receives a distinct color and its own legend entry.
    """
    rng = np.random.default_rng(
        seed
    )

    if len(
        embeddings
    ) > max_points:
        indices = rng.choice(
            len(embeddings),
            max_points,
            replace=False,
        )

        embeddings = embeddings[
            indices
        ]

        y = y[
            indices
        ]

    if len(
        embeddings
    ) < 3:
        return None

    perplexity = min(
        30,
        max(
            2,
            (
                len(
                    embeddings
                )
                - 1
            )
            // 3,
        ),
    )

    coords = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    ).fit_transform(
        embeddings
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    # Explicit discrete colors so the classes are clearly distinguishable.
    colors = plt.cm.tab20(
        np.linspace(
            0,
            1,
            max(
                len(labels),
                1,
            ),
        )
    )

    for class_idx, label in enumerate(
        labels
    ):
        selection = np.where(
            y == class_idx
        )[0]

        if len(
            selection
        ) == 0:
            continue

        ax.scatter(
            coords[
                selection,
                0,
            ],
            coords[
                selection,
                1,
            ],
            s=22,
            alpha=0.80,
            color=colors[
                class_idx
            ],
            label=(
                f"{label} "
                f"({len(selection)})"
            ),
        )

    ax.set_title(
        "GraphSAGE MFCC graph embeddings colored by genre"
    )

    ax.set_xlabel(
        "t-SNE 1"
    )

    ax.set_ylabel(
        "t-SNE 2"
    )

    ax.legend(
        fontsize=8,
        loc="best",
        framealpha=0.90,
        markerscale=1.4,
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return path


# ===========================================================================
# 9. CNN vs GNN comparison
# ===========================================================================

def save_cnn_gnn_comparison(
    gnn_val_metrics,
    out_dir,
):
    """
    Save the valid CNN-vs-GNN validation comparison.

    Important:
        CNN value = segment-level validation result from cnn_eval.ipynb.
        GNN value = graph/track-level validation result from this script.

    They are therefore shown together for reference, but the evaluation level
    is saved explicitly in JSON/CSV so the comparison is not misrepresented.
    """
    rows = [
        {
            "model": CNN_REFERENCE_NAME,
            "representation": "log-mel spectrogram",
            "evaluation_level": CNN_REFERENCE_EVAL_LEVEL,
            "split": CNN_REFERENCE_SPLIT,
            "macro_f1": float(
                CNN_VAL_MACRO_F1
            ),
            "micro_f1": float(
                CNN_VAL_MICRO_F1
            ),
        },
        {
            "model": (
                f"MFCC GraphSAGE "
                f"({EDGE_POLICY})"
            ),
            "representation": "MFCC graph",
            "evaluation_level": "track/graph-level",
            "split": "validation",
            "macro_f1": float(
                gnn_val_metrics[
                    "macro_f1"
                ]
            ),
            "micro_f1": float(
                gnn_val_metrics[
                    "micro_f1"
                ]
            ),
        },
    ]

    json_path = (
        out_dir
        / "cnn_gnn_comparison.json"
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = (
        out_dir
        / "cnn_gnn_comparison.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "representation",
                "evaluation_level",
                "split",
                "macro_f1",
                "micro_f1",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    # Side-by-side Macro-F1 / Micro-F1 comparison.
    model_names = [
        row["model"]
        for row in rows
    ]

    macro = [
        row["macro_f1"]
        for row in rows
    ]

    micro = [
        row["micro_f1"]
        for row in rows
    ]

    x = np.arange(
        len(model_names)
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars_macro = ax.bar(
        x - width / 2,
        macro,
        width,
        label="Macro-F1",
    )

    bars_micro = ax.bar(
        x + width / 2,
        micro,
        width,
        label="Micro-F1",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "CNN\n(segment-level)"
            ,
            "MFCC GraphSAGE\n(track/graph-level)"
        ]
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "F1 score"
    )

    ax.set_title(
        "CNN vs MFCC GraphSAGE validation comparison"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bars in [
        bars_macro,
        bars_micro,
    ]:
        for bar in bars:
            height = bar.get_height()

            ax.text(
                bar.get_x()
                + bar.get_width()
                / 2,
                height
                + 0.015,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.text(
        0.5,
        0.01,
        (
            "CNN score is segment-level; "
            "GNN score is track/graph-level."
        ),
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(
        rect=[
            0,
            0.04,
            1,
            1,
        ]
    )

    comparison_plot = (
        out_dir
        / "cnn_gnn_comparison.png"
    )

    fig.savefig(
        comparison_plot,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        rows,
        json_path,
        csv_path,
        comparison_plot,
    )


# ===========================================================================
# 10. Baseline
# ===========================================================================

def majority_baseline(
    train_graphs,
    test_y,
    num_classes,
):
    train_y = np.array([
        int(
            graph.y.item()
        )
        for graph
        in train_graphs
    ])

    counts = np.bincount(
        train_y,
        minlength=num_classes,
    )

    majority_class = int(
        np.argmax(
            counts
        )
    )

    pred = np.full(
        len(test_y),
        majority_class,
        dtype=int,
    )

    probs = np.zeros(
        (
            len(test_y),
            num_classes,
        ),
        dtype=np.float32,
    )

    probs[
        :,
        majority_class,
    ] = 1.0

    metrics = score(
        test_y,
        pred,
        probs,
        num_classes,
    )

    metrics[
        "majority_class"
    ] = majority_class

    return metrics


# ===========================================================================
# 11. Main training + evaluation
# ===========================================================================

def main():
    out_dir = (
        RESULT_DIR
        / RUN_NAME
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"device: {DEVICE}"
    )

    print(
        f"feature: {FEATURE}"
    )

    print(
        f"backbone: {BACKBONE}"
    )

    print(
        f"edges: {EDGE_POLICY}"
    )

    print()

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    train_graphs = load_graphs(
        "train"
    )

    val_graphs = load_graphs(
        "val"
    )

    test_graphs = load_graphs(
        "test"
    )

    space, labels = load_label_space()

    in_dim = validate_graphs(
        train_graphs,
        val_graphs,
        test_graphs,
        labels,
    )

    num_classes = len(
        labels
    )

    print(
        f"graphs: train={len(train_graphs)}, "
        f"val={len(val_graphs)}, "
        f"test={len(test_graphs)}"
    )

    print(
        f"node feature dim: {in_dim}"
    )

    print(
        f"classes: {num_classes}"
    )

    print()

    class_index = {
        str(i): label
        for i, label
        in enumerate(
            labels
        )
    }

    (
        out_dir
        / "class_index.json"
    ).write_text(
        json.dumps(
            class_index,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Loaders
    # -----------------------------------------------------------------------

    train_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    train_eval_loader = DataLoader(
        train_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_graphs,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = SingleBranchGNN(
        in_dim=in_dim,
        hidden_dim=HIDDEN_DIM,
        num_classes=num_classes,
        num_layers=NUM_LAYERS,
        backbone=BACKBONE,
    ).to(
        DEVICE
    )

    print(
        model
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    best_val_macro = 0.0
    best_val_micro = 0.0

    epochs_without_improvement = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_macro_f1": [],
        "train_micro_f1": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
        "val_auc_pr": [],
        "overfit_gap": [],
    }

    checkpoint_path = (
        out_dir
        / "best_model.pt"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
            )

            y = batch.y.reshape(
                -1
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * batch.num_graphs
            )

        train_loss = (
            total_loss
            / len(
                train_graphs
            )
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
        )

        (
            train_metrics,
            _,
            _,
            _,
        ) = evaluate(
            model,
            train_eval_loader,
            num_classes,
        )

        (
            val_metrics,
            _,
            _,
            _,
        ) = evaluate(
            model,
            val_loader,
            num_classes,
        )

        overfit_gap = (
            train_metrics[
                "macro_f1"
            ]
            - val_metrics[
                "macro_f1"
            ]
        )

        accepted = (
            val_metrics[
                "macro_f1"
            ]
            > best_val_macro
            and overfit_gap
            <= MAX_OVERFIT_GAP
        )

        rejected_spike = (
            val_metrics[
                "macro_f1"
            ]
            > best_val_macro
            and overfit_gap
            > MAX_OVERFIT_GAP
        )

        if accepted:
            best_val_macro = (
                val_metrics[
                    "macro_f1"
                ]
            )

            best_val_micro = (
                val_metrics[
                    "micro_f1"
                ]
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        history[
            "train_loss"
        ].append(
            float(
                train_loss
            )
        )

        history[
            "val_loss"
        ].append(
            float(
                val_loss
            )
        )

        history[
            "train_macro_f1"
        ].append(
            train_metrics[
                "macro_f1"
            ]
        )

        history[
            "train_micro_f1"
        ].append(
            train_metrics[
                "micro_f1"
            ]
        )

        history[
            "val_macro_f1"
        ].append(
            val_metrics[
                "macro_f1"
            ]
        )

        history[
            "val_micro_f1"
        ].append(
            val_metrics[
                "micro_f1"
            ]
        )

        history[
            "val_auc_pr"
        ].append(
            val_metrics[
                "auc_pr"
            ]
        )

        history[
            "overfit_gap"
        ].append(
            float(
                overfit_gap
            )
        )

        if accepted:
            flag = " <- saved"

        elif rejected_spike:
            flag = (
                " <- REJECTED "
                f"(gap={overfit_gap:.3f} "
                f"> {MAX_OVERFIT_GAP})"
            )

        else:
            flag = ""

        print(
            f"epoch {epoch:3d} | "
            f"loss={train_loss:.4f}/{val_loss:.4f} | "
            f"train macroF1="
            f"{train_metrics['macro_f1']:.4f} | "
            f"val macroF1="
            f"{val_metrics['macro_f1']:.4f} | "
            f"val AUC-PR="
            f"{val_metrics['auc_pr']:.4f} | "
            f"gap={overfit_gap:.3f}"
            f"{flag}"
        )

        if (
            epochs_without_improvement
            >= PATIENCE
        ):
            print(
                "\nearly stopping at "
                f"epoch {epoch} -- no accepted "
                f"improvement for {PATIENCE} epochs"
            )

            break

    # Safety fallback: if every candidate was rejected by overfit-gap rule,
    # save the final model rather than leaving no checkpoint.
    if not checkpoint_path.exists():
        print(
            "\nWARNING: no checkpoint satisfied "
            "the validation/overfit rule. "
            "Saving final epoch model."
        )

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

    (
        out_dir
        / "training_history.json"
    ).write_text(
        json.dumps(
            history,
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_loss_curve(
        history,
        out_dir
        / "loss_curve.png",
    )

    plot_f1_curve(
        history,
        out_dir
        / "f1_curve.png",
    )

    print(
        "\nbest val Macro-F1 "
        f"(accepted): {best_val_macro:.4f}"
    )

    print(
        "corresponding val Micro-F1: "
        f"{best_val_micro:.4f}"
    )

    # -----------------------------------------------------------------------
    # Load best checkpoint
    # -----------------------------------------------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    # -----------------------------------------------------------------------
    # Final evaluation
    # -----------------------------------------------------------------------

    (
        train_metrics,
        train_y,
        train_pred,
        train_probs,
    ) = evaluate(
        model,
        train_eval_loader,
        num_classes,
    )

    (
        val_metrics,
        val_y,
        val_pred,
        val_probs,
    ) = evaluate(
        model,
        val_loader,
        num_classes,
    )

    (
        test_metrics,
        test_y,
        test_pred,
        test_probs,
    ) = evaluate(
        model,
        test_loader,
        num_classes,
    )

    baseline = majority_baseline(
        train_graphs,
        test_y,
        num_classes,
    )

    # -----------------------------------------------------------------------
    # CNN comparison
    # -----------------------------------------------------------------------

    (
        cnn_gnn_comparison,
        cnn_gnn_json,
        cnn_gnn_csv,
        cnn_gnn_plot,
    ) = save_cnn_gnn_comparison(
        val_metrics,
        out_dir,
    )

    results = {
        "run": RUN_NAME,
        "feature": FEATURE,
        "backbone": BACKBONE,
        "edge_policy": EDGE_POLICY,
        "device": str(DEVICE),
        "num_classes": num_classes,
        "input_dim": in_dim,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "baseline_majority": baseline,
        "cnn_reference": {
            "source": "cnn_eval.ipynb",
            "evaluation_level": CNN_REFERENCE_EVAL_LEVEL,
            "split": CNN_REFERENCE_SPLIT,
            "macro_f1": CNN_VAL_MACRO_F1,
            "micro_f1": CNN_VAL_MICRO_F1,
        },
        "cnn_gnn_validation_comparison": cnn_gnn_comparison,
    }

    (
        out_dir
        / "test_metrics.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Per-class evaluation
    # -----------------------------------------------------------------------

    save_per_class_metrics(
        test_y,
        test_pred,
        labels,
        out_dir
        / "per_class_metrics.csv",
    )

    save_classification_report(
        test_y,
        test_pred,
        labels,
        out_dir,
    )

    cm = confusion_matrix(
        test_y,
        test_pred,
        labels=np.arange(
            num_classes
        ),
    )

    np.save(
        out_dir
        / "confusion_matrix.npy",
        cm,
    )

    plot_confusion_matrix(
        cm,
        labels,
        out_dir
        / "confusion_matrix.png",
    )

    # -----------------------------------------------------------------------
    # Test predictions
    # -----------------------------------------------------------------------

    test_track_ids = [
        str(
            graph.track_id
        )
        for graph in test_graphs
    ]

    save_predictions(
        test_y,
        test_pred,
        test_probs,
        test_track_ids,
        labels,
        out_dir
        / "test_predictions.csv",
    )

    # -----------------------------------------------------------------------
    # Embeddings + t-SNE
    # -----------------------------------------------------------------------

    (
        g_test,
        y_embedding,
        embedding_track_ids,
    ) = export_embeddings(
        model,
        test_loader,
        out_dir
        / "gnn_embeddings.pt",
    )

    tsne_path = plot_tsne(
        g_test,
        y_embedding,
        labels,
        out_dir
        / "tsne.png",
    )

    # -----------------------------------------------------------------------
    # Final report
    # -----------------------------------------------------------------------

    print(
        "\n"
        + "=" * 76
    )

    print(
        f"{'split':<20}"
        f"{'accuracy':>12}"
        f"{'Macro-F1':>12}"
        f"{'Micro-F1':>12}"
        f"{'AUC-PR':>12}"
    )

    for name, metrics in [
        (
            "majority baseline",
            baseline,
        ),
        (
            "train",
            train_metrics,
        ),
        (
            "validation",
            val_metrics,
        ),
        (
            "test",
            test_metrics,
        ),
    ]:
        print(
            f"{name:<20}"
            f"{metrics['accuracy']:>12.4f}"
            f"{metrics['macro_f1']:>12.4f}"
            f"{metrics['micro_f1']:>12.4f}"
            f"{metrics['auc_pr']:>12.4f}"
        )

    print(
        "=" * 76
    )

    print(
        "\nCNN comparison reference "
        "(validation only):"
    )

    print(
        f"  CNN segment-level       "
        f"Macro-F1={CNN_VAL_MACRO_F1:.4f} | "
        f"Micro-F1={CNN_VAL_MICRO_F1:.4f}"
    )

    print(
        f"  MFCC GraphSAGE graph-level "
        f"Macro-F1={val_metrics['macro_f1']:.4f} | "
        f"Micro-F1={val_metrics['micro_f1']:.4f}"
    )

    print(
        "  Note: evaluation levels differ; "
        "see cnn_gnn_comparison.json."
    )

    print(
        f"\nresults written to: {out_dir}"
    )

    print(
        "files:"
    )

    for path in sorted(
        out_dir.iterdir()
    ):
        print(
            f"  {path.name}"
        )

    return results


if __name__ == "__main__":
    main()
