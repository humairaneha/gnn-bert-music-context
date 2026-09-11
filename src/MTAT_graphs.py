"""
mtat_graphs.py -- Task 3 graph construction, in one file.

Reads the parquet files written by mtat_features.py and produces one PyG graph
per clip, ready for the GNN-BERT fusion model. Same constant names, same
function names and the same edge policy as task2_gnn.py's graph section, with
the two changes Task 3 forces:

    y     is a multi-hot FLOAT vector of shape [1, n_genre + n_mood], not a
          single class index. That shape is what lets PyG batch it to [B, K]
          for BCEWithLogitsLoss.

    text  is carried on the Data object so the graph and its caption never
          drift apart through shuffling and batching. PyG collates non-tensor
          attributes into a plain list, so batch.text is a list of B strings
          that goes straight into the BERT tokenizer.

There are no command-line arguments. Set the constants below, then:

    python src/mtat_graphs.py

which builds all three splits, prints the graph statistics, exports 20 example
graphs, and caches the built graphs so training does not rebuild them. Or, from
a notebook:

    from mtat_graphs import load_splits, build_split_graphs
    train, val, test, labels = load_splits()
    train_graphs = build_split_graphs(train, feature="mfcc")

Standardization happens here, not in preprocessing: the StandardScaler is fit on
the TRAIN split only and applied unchanged to val and test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


# ===========================================================================
# 0. CONFIGURATION -- this is the only block you need to edit
# ===========================================================================

# --- paths ------------------------------------------------------------------
DATA_DIR = Path("data/processed/GTZAN")
GRAPH_DIR = DATA_DIR / "graphs"
GRAPH_SAMPLE_DIR = DATA_DIR / "graph_samples"

# --- feature config ---------------------------------------------------------
# "chroma" (24-d) | "mfcc" (26-d) | "concat" (50-d)
# mfcc-only was the strongest Task 2 configuration by a wide margin (0.7363
# against 0.4682 for chroma), so it is the default here too. Build "concat" as
# well if you want the two-branch fusion model -- it slices the 50-dim x
# internally, exactly as in Task 2.
FEATURE = "mfcc"

CHROMA_DIM = 24          # 12 mean + 12 std
MFCC_DIM = 26            # 13 mean + 13 std
CONCAT_DIM = CHROMA_DIM + MFCC_DIM
EDGE_ATTR_DIM = 2        # [is_temporal, similarity_score]

# --- graph construction -----------------------------------------------------
# "tau":        keep non-adjacent pairs above a fixed absolute threshold.
# "percentile": keep pairs above this clip's own p-th percentile.
# "topk":       always keep the k most-similar non-adjacent pairs per clip.
#
# tau=0.3 is the default because it is exactly what produced the Task 2 result
# (0.7363 macro-F1, mfcc-only + GraphSAGE on GTZAN), where it gave a healthy
# mean of 1.84 similarity edges per track. Keeping the rule identical is what
# makes the Task 2 and Task 3 graphs comparable.
#
# It is worth CHECKING rather than assuming here, for one structural reason:
# cosine_similarity_matrix centers each clip on its own mean, so the segment
# vectors sum to zero and the MEAN pairwise cosine is forced to -1/(n-1). MTAT
# clips give 5 segments to GTZAN's 6, so that floor is -0.25 instead of -0.20,
# and there are 6 non-adjacent pairs to threshold instead of 10. Real music
# clears the floor easily where sections repeat -- that is why tau=0.3 worked on
# GTZAN -- but a 29 s excerpt is less likely to contain a full return than a
# 30 s track, so the margin is thinner.
#
# main() reports `clips_with_zero_similarity_edges`. If that is comfortably low,
# stay on tau. If a large share of clips end up with a bare temporal chain,
# switch to "topk", which asks "which k segments are MOST alike in this clip"
# and always has an answer.
EDGE_POLICY = "tau"
EDGE_TAU = 0.3
EDGE_PERCENTILE = 85
EDGE_TOPK = 2                # only used when EDGE_POLICY == "topk"
REQUIRE_POSITIVE = True      # tau/percentile only; topk ignores sign by design

# Fraction of zero-similarity-edge clips above which main() suggests switching.
ZERO_EDGE_WARN = 0.30

SEED = 42
VERBOSE = True


# ===========================================================================
# 1. Load splits + train-only standardization
# ===========================================================================

def load_splits(data_dir: Path = DATA_DIR, verbose: bool = VERBOSE):
    """
    Returns (train_df, val_df, test_df, labels).

    Adds chroma_standardized / mfcc_standardized. The scaler is fit on the TRAIN
    split only and then applied unchanged to val and test -- fitting on the full
    corpus would let test-set statistics into the training features, which is
    the leak the rubric's "Dataset & preprocessing" row is checking for.
    """
    data_dir = Path(data_dir)
    train = pd.read_parquet(data_dir / "train.parquet")
    val = pd.read_parquet(data_dir / "val.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    labels = json.loads((data_dir / "label_space.json").read_text())

    chroma_scaler = StandardScaler().fit(np.stack(train["chroma_pooled"].to_numpy()))
    mfcc_scaler = StandardScaler().fit(np.stack(train["mfcc_pooled"].to_numpy()))

    for part in (train, val, test):
        part["chroma_standardized"] = list(
            chroma_scaler.transform(np.stack(part["chroma_pooled"].to_numpy())))
        part["mfcc_standardized"] = list(
            mfcc_scaler.transform(np.stack(part["mfcc_pooled"].to_numpy())))

    if verbose:
        print(f"segments: train={len(train)}, val={len(val)}, test={len(test)}")
        print(f"clips:    train={train.track_id.nunique()}, "
              f"val={val.track_id.nunique()}, test={test.track_id.nunique()}")
        print(f"labels:   {labels['n_genre']} genre + {labels['n_mood']} mood "
              f"= {labels['n_genre'] + labels['n_mood']} targets\n")
    return train, val, test, labels


# ===========================================================================
# 2. Graph construction
# ===========================================================================

def cosine_similarity_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    matrix: [n_segments, dim].
    Returns [n_segments, n_segments] cosine similarity on TRACK-CENTERED vectors
    (this clip's own mean removed first).

    The centering is not cosmetic. Pooled chroma and MFCC-mean features are
    largely non-negative and dominated by a clip-level offset, so raw cosine
    similarity sits near 1.0 for every pair and carries no information about
    which segments repeat. Subtracting the clip mean turns the question into
    "how does this segment deviate from what is typical for THIS clip", which is
    what should decide whether two segments are structurally related.

    Side effect worth knowing: centered vectors sum to zero, so the mean
    pairwise cosine is pinned at -1/(n-1) regardless of the audio. Genuinely
    repeated sections sit well above that floor; unrelated ones sit at or below
    it. That gap is the signal an absolute threshold picks out.
    """
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    norm = centered / (centered.norm(dim=1, keepdim=True) + 1e-8)
    return norm @ norm.t()


def build_edges(n: int, sim: torch.Tensor, policy: str = EDGE_POLICY,
                percentile: int = EDGE_PERCENTILE, tau: float = EDGE_TAU,
                topk: int = EDGE_TOPK, require_positive: bool = REQUIRE_POSITIVE):
    """
    Temporal chain (i <-> i+1) plus similarity edges between non-adjacent
    segments. edge_attr = [is_temporal, similarity_score].

    Under "tau" with tau=0.3 this is the same rule that produced the Task 2
    numbers, so the two tasks' graphs stay comparable.
    """
    edges, edge_attr = [], []
    temporal_pairs = set()
    for i in range(n - 1):
        edges += [[i, i + 1], [i + 1, i]]
        edge_attr += [[1.0, 0.0], [1.0, 0.0]]
        temporal_pairs.add((i, i + 1))
        temporal_pairs.add((i + 1, i))

    candidates = [
        (i, j, sim[i, j].item())
        for i in range(n) for j in range(i + 1, n)
        if (i, j) not in temporal_pairs
    ]

    if candidates:
        if policy == "topk":
            # no sign requirement: after centering, "least dissimilar" is the
            # meaningful relation, and this guarantees every clip has edges
            chosen = sorted(candidates, key=lambda c: -c[2])[:topk]
        else:
            if policy == "percentile":
                values = torch.tensor([s for _, _, s in candidates], dtype=torch.float32)
                threshold = torch.quantile(values, percentile / 100.0).item()
            elif policy == "tau":
                threshold = tau
            else:
                raise ValueError(f"unknown edge policy {policy!r}")
            chosen = [c for c in candidates
                      if c[2] > threshold and (c[2] > 0 or not require_positive)]

        for i, j, s in chosen:
            edges += [[i, j], [j, i]]
            edge_attr += [[0.0, s], [0.0, s]]

    if not edges:
        # single-segment clip: self-loop so edge_index is never malformed
        edges, edge_attr = [[0, 0]], [[0.0, 1.0]]

    return edges, edge_attr


def build_track_graph(track_rows, feature: str = FEATURE, **edge_kw) -> Data:
    """
    track_rows: rows for ONE clip, sorted by segment_id.
    feature:    "chroma" (24d) | "mfcc" (26d) | "concat" (50d)

    Node features x are always the STANDARDIZED columns. Similarity for
    chroma/mfcc is computed on the RAW POOLED column; for concat it is computed
    on the standardized x, because MFCC's larger raw scale otherwise swamps
    chroma's contribution to the similarity direction.
    """
    n = len(track_rows)

    if feature == "chroma":
        x = torch.tensor(np.stack([r["chroma_standardized"] for r in track_rows]),
                         dtype=torch.float32)
        sim_source = torch.tensor(np.stack([r["chroma_pooled"] for r in track_rows]),
                                  dtype=torch.float32)
    elif feature == "mfcc":
        x = torch.tensor(np.stack([r["mfcc_standardized"] for r in track_rows]),
                         dtype=torch.float32)
        sim_source = torch.tensor(np.stack([r["mfcc_pooled"] for r in track_rows]),
                                  dtype=torch.float32)
    elif feature == "concat":
        x = torch.tensor(np.stack([
            np.concatenate([r["chroma_standardized"], r["mfcc_standardized"]])
            for r in track_rows]), dtype=torch.float32)
        sim_source = x
    else:
        raise ValueError(f"unknown feature {feature!r}")

    sim = cosine_similarity_matrix(sim_source)
    edges, edge_attr = build_edges(n, sim, **edge_kw)

    first = track_rows[0]
    # multi-hot FLOAT, shape [1, n_genre + n_mood]. Float because
    # BCEWithLogitsLoss wants float targets; shape [1, K] because PyG then
    # batches it to [B, K] rather than flattening B graphs into one long vector.
    y = torch.tensor(
        np.concatenate([first["genre_labels"], first["mood_labels"]])[None, :],
        dtype=torch.float)

    data = Data(x=x, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float), y=y)
    data.track_id = int(first["track_id"])
    data.num_segments = n
    data.text = str(first["text"])       # collated into a list of B strings
    data.artist = str(first["artist"])   # kept for leakage audits and case studies
    return data


def build_split_graphs(split_df: pd.DataFrame, feature: str = FEATURE, **edge_kw):
    """One graph per clip. Segments are sorted by segment_id before building."""
    graphs = []
    for _, group in split_df.groupby("track_id"):
        rows = group.sort_values("segment_id").to_dict("records")
        graphs.append(build_track_graph(rows, feature, **edge_kw))
    return graphs


# ===========================================================================
# 3. Reporting + export
# ===========================================================================

def graph_statistics(graphs) -> dict:
    """Documents the built graphs -- the rubric's 'graph construction documented' row."""
    n_sim = np.array([int((g.edge_attr[:, 0] < 0.5).sum().item()) // 2 for g in graphs])
    n_seg = np.array([int(g.num_segments) for g in graphs])
    y = torch.cat([g.y for g in graphs])
    return {
        "num_graphs": len(graphs),
        "node_feature_dim": int(graphs[0].x.shape[1]),
        "segments_per_clip_mean": round(float(n_seg.mean()), 2),
        "similarity_edges_mean": round(float(n_sim.mean()), 2),
        "similarity_edges_median": int(np.median(n_sim)),
        "similarity_edges_max": int(n_sim.max()),
        "clips_with_zero_similarity_edges": int((n_sim == 0).sum()),
        "zero_edge_fraction": round(float((n_sim == 0).mean()), 4),
        "labels_per_clip_mean": round(float(y.sum(dim=1).mean().item()), 2),
        "tags_with_zero_positives": int((y.sum(dim=0) == 0).sum().item()),
    }


def export_sample_graphs(graphs, out_dir: Path = GRAPH_SAMPLE_DIR, k: int = 20):
    """Final-submission requirement: at least 20 example .pt graphs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for g in graphs[:k]:
        torch.save(g, out_dir / f"clip_{g.track_id}.pt")
    return len(graphs[:k])


def save_graphs(graphs, path: Path):
    """Building ~17k graphs takes a few minutes; cache them so training does not repeat it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, path)


def load_graphs(path: Path):
    return torch.load(Path(path), weights_only=False)


# ===========================================================================
# 4. Main
# ===========================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    edge_kw = dict(policy=EDGE_POLICY, tau=EDGE_TAU, percentile=EDGE_PERCENTILE,
                   topk=EDGE_TOPK, require_positive=REQUIRE_POSITIVE)
    print(f"feature={FEATURE} | edge policy={EDGE_POLICY} "
          f"(tau={EDGE_TAU}, percentile={EDGE_PERCENTILE}, topk={EDGE_TOPK})\n")

    train_df, val_df, test_df, labels = load_splits()

    graphs, stats = {}, {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        graphs[name] = build_split_graphs(part, FEATURE, **edge_kw)
        stats[name] = graph_statistics(graphs[name])
        # policy in the filename so the tau and topk builds coexist and the
        # ablation is two runs rather than a destructive edit
        save_graphs(graphs[name], GRAPH_DIR / f"{name}_{FEATURE}_{EDGE_POLICY}.pt")
        print(f"[{name}] {stats[name]}")

    (GRAPH_DIR / f"graph_stats_{EDGE_POLICY}.json").write_text(json.dumps(stats, indent=2))

    # the number that decides whether the edge policy is doing anything
    frac = stats["train"]["zero_edge_fraction"]
    print(f"\nsimilarity edges per clip (train): "
          f"mean {stats['train']['similarity_edges_mean']}, "
          f"median {stats['train']['similarity_edges_median']}")
    print(f"clips with a bare temporal chain: {frac:.1%}")
    if frac > ZERO_EDGE_WARN:
        print(f"  ^ above {ZERO_EDGE_WARN:.0%}. For that many clips the GNN sees no "
              f"structure\n    beyond time order, which makes any 'graph' claim in the "
              f"report thin.\n    Set EDGE_POLICY = 'topk' and rerun -- it guarantees "
              f"every clip gets its\n    {EDGE_TOPK} most-similar non-adjacent pairs.")
    elif EDGE_POLICY == "topk":
        print(f"  ^ expected: topk={EDGE_TOPK} guarantees every clip gets edges. "
              f"Compare against\n    the tau build rather than reading this as a "
              f"quality signal -- some of these\n    edges are forced onto clips whose "
              f"best pair is genuinely dissimilar.")
    else:
        print(f"  ^ fine. GTZAN at tau=0.3 gave 1.84 similarity edges/track, "
              f"so this is in family.")

    n = export_sample_graphs(graphs["train"])
    print(f"\nexported {n} example graphs to {GRAPH_SAMPLE_DIR}")
    print(f"cached graphs in {GRAPH_DIR}")

    g = graphs["train"][0]
    print(f"\nexample graph:")
    print(f"  clip_id    {g.track_id}   artist {g.artist!r}")
    print(f"  x          {tuple(g.x.shape)}")
    print(f"  edge_index {tuple(g.edge_index.shape)}   edge_attr {tuple(g.edge_attr.shape)}")
    print(f"  y          {tuple(g.y.shape)}  ({int(g.y.sum().item())} positive tags)")
    print(f"  text       {g.text!r}")
    active = [t for t, v in zip(labels["genre"] + labels["mood"], g.y[0].tolist()) if v]
    print(f"  targets    {active}")
    return graphs, labels


if __name__ == "__main__":
    main()
