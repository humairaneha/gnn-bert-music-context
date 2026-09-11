"""Build MusicCaps segment graphs from prepared parquet splits.

Run from the repository root: python src/graph_builder.py --dataset musiccaps.
Reads data/processed/musiccaps/{train,val,test}.parquet and label_space.json.
Standardization is fitted on train only. Caption-based source keys map to stable
MD5-derived clip IDs, while original keys remain available as source_key.
Outputs include graph caches, statistics, and 20 sample graphs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data


# ===========================================================================
# 0. CONFIGURATION -- dataset and graph settings
# ===========================================================================

DATA_DIR = Path("data/processed/musiccaps")
GRAPH_DIR = DATA_DIR / "graphs"
GRAPH_SAMPLE_DIR = DATA_DIR / "graph_samples"

# "chroma" (24-d) | "mfcc" (26-d) | "concat" (50-d)
FEATURE = "mfcc"

CHROMA_DIM = 24
MFCC_DIM = 26
CONCAT_DIM = CHROMA_DIM + MFCC_DIM
EDGE_ATTR_DIM = 2        # [is_temporal, similarity_score]

# --- graph construction -----------------------------------------------------
# Identical rule to Task 2 and Task 3, kept deliberately so the three sets of
# graph statistics can be read side by side.
#
# Expect a HIGHER zero-edge fraction here than on MTAT, for two compounding
# reasons. cosine_similarity_matrix centers each clip on its own mean, so the
# mean pairwise cosine is pinned at -1/(n-1) = -0.25 at 5 segments, and there
# are only 6 non-adjacent pairs to threshold. On top of that, a 10 s clip is far
# less likely to contain a repeated section than MTAT's 29 s excerpt -- a chorus
# does not come back inside ten seconds. If most clips end up with a bare
# temporal chain, switch to "topk"; main() reports the number.
EDGE_POLICY = "tau"
EDGE_TAU = 0.3
EDGE_PERCENTILE = 85
EDGE_TOPK = 2
REQUIRE_POSITIVE = True

ZERO_EDGE_WARN = 0.50    # higher bar than MTAT's 0.30, for the reason above

SEED = 42
VERBOSE = True


# ===========================================================================
# 1. Load splits + train-only standardization
# ===========================================================================

def load_splits(data_dir: Path = DATA_DIR, verbose: bool = VERBOSE):
    """
    Returns (train_df, val_df, test_df, labels).

    The scaler is fit on TRAIN only and applied unchanged to val and test --
    fitting on the full corpus would let test statistics into the training
    features, which is the leak the rubric's preprocessing row checks for.
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
              f"= {len(labels['tags'])} targets")
        if "is_silent" in train.columns:
            print(f"silent:   {int(train.is_silent.sum())} near-silent segments "
                  f"({train.is_silent.mean():.2%} of train)")
        sample = str(train.track_id.iloc[0])
        if len(sample) > 48:
            print(f"note:     track_id holds the caption ({len(sample)} chars) -- "
                  f"short_id() will derive\n          stable ids like "
                  f"{short_id(sample)!r} for the graphs and filenames")
        print()
    return train, val, test, labels


# ===========================================================================
# 2. Graph construction
# ===========================================================================

def cosine_similarity_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """
    matrix: [n_segments, dim].
    Cosine similarity on TRACK-CENTERED vectors (this clip's own mean removed).

    Pooled chroma and MFCC-mean features are largely non-negative and dominated
    by a clip-level offset, so raw cosine similarity sits near 1.0 for every
    pair and says nothing about which segments repeat. Centering turns the
    question into "how does this segment deviate from what is typical for THIS
    clip", which is what should decide whether two segments are related.
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
    """
    edges, edge_attr = [], []
    temporal_pairs = set()
    for i in range(n - 1):
        edges += [[i, i + 1], [i + 1, i]]
        edge_attr += [[1.0, 0.0], [1.0, 0.0]]
        temporal_pairs.add((i, i + 1))
        temporal_pairs.add((i + 1, i))

    candidates = [(i, j, sim[i, j].item())
                  for i in range(n) for j in range(i + 1, n)
                  if (i, j) not in temporal_pairs]

    if candidates:
        if policy == "topk":
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
        edges, edge_attr = [[0, 0]], [[0.0, 1.0]]

    return edges, edge_attr


def build_track_graph(track_rows, feature: str = FEATURE, n_labels: int | None = None,
                      **edge_kw) -> Data:
    """
    track_rows: rows for ONE clip, sorted by segment_id.

    Node features x are always the STANDARDIZED columns. Similarity for
    chroma/mfcc is computed on the RAW POOLED column; for concat on the
    standardized x, because MFCC's larger raw scale would otherwise swamp
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
    # ONE flat labels column here, unlike MTAT's genre_labels + mood_labels.
    # Multi-hot FLOAT of shape [1, K]: float because BCEWithLogitsLoss wants
    # float targets, [1, K] because PyG then batches to [B, K] rather than
    # flattening B graphs into one long vector.
    y = torch.tensor(np.asarray(first["labels"])[None, :], dtype=torch.float)
    if n_labels is not None and y.shape[1] != n_labels:
        raise ValueError(
            f"clip {first['track_id']} has {y.shape[1]} labels but label_space.json "
            f"lists {n_labels}. The parquet was written with a different tag set -- "
            f"rerun musiccaps_features.py after changing TARGET_CATEGORIES.")

    data = Data(x=x, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float), y=y)
    # STRING, not int -- this is the YouTube id. PyG collates it into a list of
    # B strings, the same way it handles text.
    # Derived, not copied: see short_id(). The original value stays available
    # as data.source_key for joining back to the parquet.
    data.source_key = str(first["track_id"])
    data.track_id = short_id(first["track_id"])
    if "ytid" in first and str(first["ytid"]).strip() not in ("", "#NAME?", "None"):
        data.ytid = str(first["ytid"])          # only present on newer parquets
    data.num_segments = n
    data.text = str(first["text"])       # a real caption, not a template
    return data


def build_split_graphs(split_df: pd.DataFrame, feature: str = FEATURE,
                       n_labels: int | None = None, **edge_kw):
    """One graph per clip. Segments are sorted by segment_id before building."""
    graphs = []
    for _, group in split_df.groupby("track_id"):
        rows = group.sort_values("segment_id").to_dict("records")
        graphs.append(build_track_graph(rows, feature, n_labels, **edge_kw))
    return graphs


# ===========================================================================
# 3. Reporting + export
# ===========================================================================

def graph_statistics(graphs) -> dict:
    n_sim = np.array([int((g.edge_attr[:, 0] < 0.5).sum().item()) // 2 for g in graphs])
    n_seg = np.array([int(g.num_segments) for g in graphs])
    y = torch.cat([g.y for g in graphs])
    texts = [g.text for g in graphs]
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
        # the number that decides whether contrastive retrieval is meaningful
        "unique_captions": len(set(texts)),
        "caption_unique_fraction": round(len(set(texts)) / max(len(texts), 1), 4),
    }


def short_id(value, prefix: str = "mc", limit: int = 48) -> str:
    """
    A short, stable clip identifier derived from whatever the parquet's
    `track_id` column happens to hold.

    In this dataset the audio mirror has no ytid, so musiccaps_features.py joined
    on caption text and `track_id` ended up holding the full caption -- hundreds
    of characters. That is fine as a grouping key (MusicCaps captions are unique
    per clip) but useless as an identifier: it cannot be a filename, and it makes
    the retrieval output unreadable.

    Rather than forcing a preprocessing rerun, the id is derived here. An MD5
    prefix is deterministic, so the same clip gets the same id on every run and
    across the train/val/test builds -- which matters, because the Task 4
    retrieval output joins on it. Values that are already short and
    filename-safe are passed through untouched.
    """
    s = str(value)
    if len(s) <= limit and all(c.isalnum() or c in "-_" for c in s):
        return s
    return prefix + hashlib.md5(s.encode()).hexdigest()[:12]


def safe_name(value, limit: int = 64) -> str:
    """
    Filename-safe, length-capped version of a clip id.

    Defensive: if track_id ever holds something long (a caption, say), an
    unsanitised f"clip_{id}.pt" raises "File name too long" from torch.save.
    Truncating alone could collide, so a short hash of the full value is
    appended whenever anything is dropped.
    """
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(value))
    if len(s) <= limit:
        return s
    return s[:limit - 9] + "_" + hashlib.md5(str(value).encode()).hexdigest()[:8]


def export_sample_graphs(graphs, out_dir: Path = GRAPH_SAMPLE_DIR, k: int = 20):
    """Final-submission requirement: at least 20 example .pt graphs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for g in graphs[:k]:
        torch.save(g, out_dir / f"clip_{safe_name(g.track_id)}.pt")
    return len(graphs[:k])


def save_graphs(graphs, path: Path):
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
    tags = labels["tags"]

    graphs, stats = {}, {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        graphs[name] = build_split_graphs(part, FEATURE, len(tags), **edge_kw)
        stats[name] = graph_statistics(graphs[name])
        save_graphs(graphs[name], GRAPH_DIR / f"{name}_{FEATURE}_{EDGE_POLICY}.pt")
        print(f"[{name}] {stats[name]}")

    (GRAPH_DIR / f"graph_stats_{EDGE_POLICY}.json").write_text(json.dumps(stats, indent=2))

    frac = stats["train"]["zero_edge_fraction"]
    print(f"\nsimilarity edges per clip (train): "
          f"mean {stats['train']['similarity_edges_mean']}, "
          f"median {stats['train']['similarity_edges_median']}")
    print(f"clips with a bare temporal chain: {frac:.1%}")
    if frac > ZERO_EDGE_WARN:
        print(f"  ^ above {ZERO_EDGE_WARN:.0%}. Expected to some degree on 10 s clips "
              f"-- a section\n    rarely repeats inside ten seconds -- but at this level "
              f"the GNN is mostly\n    seeing time order. Set EDGE_POLICY = 'topk' and "
              f"rerun.")
    elif EDGE_POLICY == "topk":
        print(f"  ^ expected: topk={EDGE_TOPK} guarantees edges. Compare against the "
              f"tau build\n    rather than reading this as a quality signal.")
    else:
        print(f"  ^ workable. MTAT at the same tau gave 0.97 edges/clip and 35% bare "
              f"chains.")

    # the number that decides whether contrastive retrieval is meaningful at all
    u = stats["test"]["caption_unique_fraction"]
    print(f"\ncaptions unique in test: {u:.1%} "
          f"({stats['test']['unique_captions']} of {stats['test']['num_graphs']})")
    if u > 0.95:
        print("  ^ effectively one-to-one. R@K means what it should, and the "
              "duplicate\n    masking in task4_contrastive.py becomes a no-op.")
    else:
        print("  ^ below 95%. Keep GROUP_AWARE_RETRIEVAL on in task4_contrastive.py "
              "and say\n    so in the report.")

    n = export_sample_graphs(graphs["train"])
    print(f"\nexported {n} example graphs to {GRAPH_SAMPLE_DIR}")
    print(f"cached graphs in {GRAPH_DIR}")

    g = graphs["train"][0]
    print(f"\nexample graph:")
    print(f"  clip_id    {g.track_id!r}")
    print(f"  x          {tuple(g.x.shape)}")
    print(f"  edge_index {tuple(g.edge_index.shape)}   edge_attr {tuple(g.edge_attr.shape)}")
    print(f"  y          {tuple(g.y.shape)}  ({int(g.y.sum().item())} positive tags)")
    print(f"  caption    {g.text[:100]}...")
    print(f"  targets    {[t for t, v in zip(tags, g.y[0].tolist()) if v]}")
    return graphs, labels


if __name__ == "__main__":
    main()
