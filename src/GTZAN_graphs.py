"""Build GTZAN segment graphs from prepared MFCC feature splits.

Inputs: data/processed/GTZAN/{train,val,test}.parquet and label_space.json.
Each node contains a standardized 26-dimensional pooled MFCC vector. Temporal
edges join adjacent segments; the configured similarity policy adds links
between non-adjacent segments. Edge attributes are
[is_temporal, similarity_score]. Each graph has one integer genre target.

Graph caches and statistics are written to data/processed/GTZAN/graphs/.
Individual training examples are exported to its sibling graph_samples/.

    python src/graph_builder.py --dataset gtzan
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
# 0. CONFIGURATION -- edit only this block
# ===========================================================================

DATA_DIR = Path("data/processed/GTZAN")

GRAPH_DIR = DATA_DIR / "graphs"
GRAPH_SAMPLE_DIR = DATA_DIR / "graph_samples"

# Edge construction policy:
#   "tau"        -> keep non-adjacent pairs with similarity > EDGE_TAU
#   "percentile" -> keep non-adjacent pairs above each track's percentile threshold
#   "topk"       -> keep the EDGE_TOPK most similar non-adjacent pairs
EDGE_POLICY = "tau"

EDGE_TAU = 0.3
EDGE_PERCENTILE = 85
EDGE_TOPK = 2

# Used by tau/percentile. topk ignores this.
REQUIRE_POSITIVE = True

N_SAMPLE_GRAPHS = 20

SEED = 42
VERBOSE = True


# ===========================================================================
# 1. Load splits + train-only MFCC standardization
# ===========================================================================

def load_splits(data_dir: Path = DATA_DIR, verbose: bool = VERBOSE):
    """
    Load train/val/test parquet files and label_space.json.

    StandardScaler is fitted ONLY on train MFCC pooled features and then
    applied unchanged to train, val, and test.
    """
    data_dir = Path(data_dir)

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    test_path = data_dir / "test.parquet"
    label_path = data_dir / "label_space.json"

    for path in [train_path, val_path, test_path, label_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    labels = json.loads(label_path.read_text(encoding="utf-8"))

    required = {
        "track_id",
        "segment_id",
        "genre",
        "mfcc_pooled",
    }

    for name, part in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        missing = required - set(part.columns)
        if missing:
            raise KeyError(
                f"{name}.parquet is missing required columns: "
                f"{sorted(missing)}"
            )

    # Fit ONLY on train split.
    train_mfcc = np.stack(train["mfcc_pooled"].to_numpy())

    mfcc_scaler = StandardScaler().fit(train_mfcc)

    # Apply the same fitted scaler to all splits.
    for part in [train, val, test]:
        part["mfcc_standardized"] = list(
            mfcc_scaler.transform(
                np.stack(part["mfcc_pooled"].to_numpy())
            )
        )

    # Use label_space.json as the source of class ordering.
    genres = labels.get("genre", [])

    if not genres:
        genres = labels.get("tags", [])

    if not genres:
        raise ValueError(
            "label_space.json does not contain a usable `genre` or `tags` list."
        )

    label_map = {
        genre: i
        for i, genre in enumerate(genres)
    }

    # Ensure parquet genres exist in label_space.json.
    parquet_genres = (
        set(train["genre"].tolist())
        | set(val["genre"].tolist())
        | set(test["genre"].tolist())
    )

    missing_genres = [
        genre
        for genre in parquet_genres
        if genre not in label_map
    ]

    if missing_genres:
        raise ValueError(
            "Some parquet genres are missing from label_space.json: "
            f"{missing_genres}"
        )

    if verbose:
        print(
            f"segments: train={len(train)}, "
            f"val={len(val)}, test={len(test)}"
        )

        print(
            f"tracks:   train={train.track_id.nunique()}, "
            f"val={val.track_id.nunique()}, "
            f"test={test.track_id.nunique()}"
        )

        print(f"genres:   {len(genres)}")
        print(f"MFCC dim: {train_mfcc.shape[1]}")
        print()

    return train, val, test, labels, label_map


# ===========================================================================
# 2. Graph construction
# ===========================================================================

def cosine_similarity_matrix(pooled_matrix):
    """
    pooled_matrix: [n_segments, 26]

    Returns track-centered cosine similarity matrix.
    """
    track_mean = pooled_matrix.mean(
        dim=0,
        keepdim=True,
    )

    centered = pooled_matrix - track_mean

    norm = centered / (
        centered.norm(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )

    sim = norm @ norm.t()

    return sim


def build_track_graph(
    track_rows,
    label_map,
    policy=EDGE_POLICY,
    tau=EDGE_TAU,
    percentile=EDGE_PERCENTILE,
    topk=EDGE_TOPK,
    require_positive=REQUIRE_POSITIVE,
):
    """
    Build one PyG graph for one track.

    Node features:
        standardized MFCC pooled features

    Similarity source:
        raw MFCC pooled features

    Target:
        scalar integer genre class
    """
    n = len(track_rows)

    pooled_matrix = torch.tensor(
        [
            row["mfcc_pooled"]
            for row in track_rows
        ],
        dtype=torch.float32,
    )

    x = torch.tensor(
        [
            row["mfcc_standardized"]
            for row in track_rows
        ],
        dtype=torch.float32,
    )

    sim = cosine_similarity_matrix(
        pooled_matrix
    )

    edges = []
    edge_attr = []

    temporal_pairs = set()

    # Temporal edges
    for i in range(n - 1):
        edges += [
            [i, i + 1],
            [i + 1, i],
        ]

        edge_attr += [
            [1.0, 0.0],
            [1.0, 0.0],
        ]

        temporal_pairs.add(
            (i, i + 1)
        )

        temporal_pairs.add(
            (i + 1, i)
        )

    # Non-adjacent similarity candidates
    candidates = [
        (
            i,
            j,
            sim[i, j].item(),
        )
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in temporal_pairs
    ]

    # Configurable similarity-edge rule
    chosen = []

    if candidates:
        if policy == "topk":
            chosen = sorted(
                candidates,
                key=lambda item: -item[2],
            )[:topk]

        elif policy == "percentile":
            values = torch.tensor(
                [score for _, _, score in candidates],
                dtype=torch.float32,
            )

            threshold = torch.quantile(
                values,
                percentile / 100.0,
            ).item()

            chosen = [
                (i, j, score)
                for i, j, score in candidates
                if score > threshold
                and (
                    score > 0
                    or not require_positive
                )
            ]

        elif policy == "tau":
            chosen = [
                (i, j, score)
                for i, j, score in candidates
                if score > tau
                and (
                    score > 0
                    or not require_positive
                )
            ]

        else:
            raise ValueError(
                f"Unknown edge policy: {policy!r}. "
                "Use 'tau', 'percentile', or 'topk'."
            )

    for i, j, score in chosen:
        edges += [
            [i, j],
            [j, i],
        ]

        edge_attr += [
            [0.0, score],
            [0.0, score],
        ]

    # Single-node graph fallback
    if len(edges) == 0:
        edges = [
            [0, 0]
        ]

        edge_attr = [
            [0.0, 1.0]
        ]

    edge_index = torch.tensor(
        edges,
        dtype=torch.long,
    ).t().contiguous()

    edge_attr = torch.tensor(
        edge_attr,
        dtype=torch.float32,
    )

    y = torch.tensor(
        [
            label_map[
                track_rows[0]["genre"]
            ]
        ],
        dtype=torch.long,
    )

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
    )

    data.track_id = track_rows[0]["track_id"]
    data.num_segments = n

    return data


def build_split_graphs(
    split_df: pd.DataFrame,
    label_map,
    policy=EDGE_POLICY,
    tau=EDGE_TAU,
    percentile=EDGE_PERCENTILE,
    topk=EDGE_TOPK,
    require_positive=REQUIRE_POSITIVE,
):
    """
    Build one graph per track_id.

    Segments are sorted by segment_id before graph creation.
    """
    graphs = []

    for track_id, group in split_df.groupby(
        "track_id"
    ):
        group = group.sort_values(
            "segment_id"
        )

        track_rows = group.to_dict(
            "records"
        )

        graph = build_track_graph(
            track_rows,
            label_map,
            policy=policy,
            tau=tau,
            percentile=percentile,
            topk=topk,
            require_positive=require_positive,
        )

        graphs.append(graph)

    return graphs


# ===========================================================================
# 3. Graph statistics
# ===========================================================================

def graph_statistics(graphs):
    """
    Summarize constructed graph structure.
    """
    if not graphs:
        return {
            "num_graphs": 0,
            "node_feature_dim": 0,
            "segments_per_track_mean": 0.0,
            "similarity_edges_mean": 0.0,
            "similarity_edges_median": 0,
            "similarity_edges_max": 0,
            "tracks_with_zero_similarity_edges": 0,
            "zero_similarity_edge_fraction": 0.0,
        }

    similarity_edges = np.array([
        int(
            (
                graph.edge_attr[:, 0] < 0.5
            ).sum().item()
        ) // 2
        for graph in graphs
    ])

    segments = np.array([
        int(graph.num_segments)
        for graph in graphs
    ])

    return {
        "num_graphs":
            len(graphs),

        "node_feature_dim":
            int(graphs[0].x.shape[1]),

        "segments_per_track_mean":
            round(
                float(
                    segments.mean()
                ),
                2,
            ),

        "similarity_edges_mean":
            round(
                float(
                    similarity_edges.mean()
                ),
                2,
            ),

        "similarity_edges_median":
            int(
                np.median(
                    similarity_edges
                )
            ),

        "similarity_edges_max":
            int(
                similarity_edges.max()
            ),

        "tracks_with_zero_similarity_edges":
            int(
                (
                    similarity_edges == 0
                ).sum()
            ),

        "zero_similarity_edge_fraction":
            round(
                float(
                    (
                        similarity_edges == 0
                    ).mean()
                ),
                4,
            ),
    }


# ===========================================================================
# 4. Save / load graphs
# ===========================================================================

def save_graphs(graphs, path):
    """
    Save the complete graph list for one split.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        graphs,
        path,
    )


def load_graphs(path):
    """
    Load a previously saved graph list.
    """
    return torch.load(
        Path(path),
        weights_only=False,
    )


def export_sample_graphs(
    graphs,
    out_dir=GRAPH_SAMPLE_DIR,
    k=N_SAMPLE_GRAPHS,
):
    """
    Save the first k training graphs as individual .pt files.
    """
    out_dir = Path(out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    n = min(
        k,
        len(graphs),
    )

    for i, graph in enumerate(
        graphs[:n]
    ):
        track_id = str(
            graph.track_id
        ).replace(
            "/",
            "_",
        )

        torch.save(
            graph,
            out_dir
            / f"graph_{i:02d}_track_{track_id}.pt",
        )

    return n


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(
        f"MFCC graph creation | "
        f"edge_policy={EDGE_POLICY} | "
        f"tau={EDGE_TAU} | "
        f"percentile={EDGE_PERCENTILE} | "
        f"topk={EDGE_TOPK}"
    )

    print()

    (
        train_df,
        val_df,
        test_df,
        labels,
        label_map,
    ) = load_splits()

    print("Building train graphs...")

    train_graphs = build_split_graphs(
        train_df,
        label_map,
        policy=EDGE_POLICY,
        tau=EDGE_TAU,
        percentile=EDGE_PERCENTILE,
        topk=EDGE_TOPK,
        require_positive=REQUIRE_POSITIVE,
    )

    print("Building validation graphs...")

    val_graphs = build_split_graphs(
        val_df,
        label_map,
        policy=EDGE_POLICY,
        tau=EDGE_TAU,
        percentile=EDGE_PERCENTILE,
        topk=EDGE_TOPK,
        require_positive=REQUIRE_POSITIVE,
    )

    print("Building test graphs...")

    test_graphs = build_split_graphs(
        test_df,
        label_map,
        policy=EDGE_POLICY,
        tau=EDGE_TAU,
        percentile=EDGE_PERCENTILE,
        topk=EDGE_TOPK,
        require_positive=REQUIRE_POSITIVE,
    )

    # -----------------------------------------------------------------------
    # Save complete graph datasets
    # -----------------------------------------------------------------------

    train_path = (
        GRAPH_DIR
        / f"train_mfcc_{EDGE_POLICY}.pt"
    )

    val_path = (
        GRAPH_DIR
        / f"val_mfcc_{EDGE_POLICY}.pt"
    )

    test_path = (
        GRAPH_DIR
        / f"test_mfcc_{EDGE_POLICY}.pt"
    )

    save_graphs(
        train_graphs,
        train_path,
    )

    save_graphs(
        val_graphs,
        val_path,
    )

    save_graphs(
        test_graphs,
        test_path,
    )

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    stats = {
        "train":
            graph_statistics(
                train_graphs
            ),

        "val":
            graph_statistics(
                val_graphs
            ),

        "test":
            graph_statistics(
                test_graphs
            ),
    }

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats_path = (
        GRAPH_DIR
        / f"graph_stats_{EDGE_POLICY}.json"
    )

    stats_path.write_text(
        json.dumps(
            stats,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Export 20 sample graphs
    # -----------------------------------------------------------------------

    n_exported = export_sample_graphs(
        train_graphs,
        GRAPH_SAMPLE_DIR,
        N_SAMPLE_GRAPHS,
    )

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------

    print()

    print(
        f"[train] {stats['train']}"
    )

    print(
        f"[val]   {stats['val']}"
    )

    print(
        f"[test]  {stats['test']}"
    )

    print()

    print(
        "Saved complete graph datasets:"
    )

    print(
        f"  {train_path}"
    )

    print(
        f"  {val_path}"
    )

    print(
        f"  {test_path}"
    )

    print()

    print(
        f"Saved graph statistics: "
        f"{stats_path}"
    )

    print(
        f"Exported {n_exported} sample graphs to: "
        f"{GRAPH_SAMPLE_DIR}"
    )

    # -----------------------------------------------------------------------
    # Example graph
    # -----------------------------------------------------------------------

    if train_graphs:
        graph = train_graphs[0]

        inv_label_map = {
            value: key
            for key, value
            in label_map.items()
        }

        genre = inv_label_map[
            int(
                graph.y.item()
            )
        ]

        n_similarity = (
            int(
                (
                    graph.edge_attr[:, 0]
                    < 0.5
                ).sum().item()
            )
            // 2
        )

        print()

        print("Example graph:")

        print(
            f"  track_id:    "
            f"{graph.track_id}"
        )

        print(
            f"  x:           "
            f"{tuple(graph.x.shape)}"
        )

        print(
            f"  edge_index:  "
            f"{tuple(graph.edge_index.shape)}"
        )

        print(
            f"  edge_attr:   "
            f"{tuple(graph.edge_attr.shape)}"
        )

        print(
            f"  y:           "
            f"{tuple(graph.y.shape)}"
        )

        print(
            f"  genre:       "
            f"{genre}"
        )

        print(
            f"  segments:    "
            f"{graph.num_segments}"
        )

        print(
            f"  similarity edges: "
            f"{n_similarity}"
        )

    return (
        train_graphs,
        val_graphs,
        test_graphs,
        labels,
    )


if __name__ == "__main__":
    main()
