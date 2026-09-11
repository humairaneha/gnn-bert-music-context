"""
gnn_model.py -- GNN encoders, public model interfaces.

Task 3 / Task 4 (multi-label, MagnaTagATune and MusicCaps):
    GraphSAGE          three SAGEConv layers, mean-pool readout, multi-label head

Task 2 (single-label genre, GTZAN):
    GraphSAGEEncoder   ignores edge_attr; SAGEConv has no mechanism for it
    GATEncoder         uses edge_attr via edge_dim, so attention can condition on
                       whether an edge is temporal or similarity-based
    SingleBranchGNN    chroma-only (24-d) / mfcc-only (26-d) / concat (50-d)
    TwoBranchFusionGNN separate encoders per modality, fused after pooling

The implementations live in gnn.py and GTZAN_gnn.py. This module only re-exports
them, so there is exactly one definition of each model in the repository.
"""

from gnn import GraphSAGE

try:
    from GTZAN_gnn import (
        GraphSAGEEncoder,
        GATEncoder,
        SingleBranchGNN,
        TwoBranchFusionGNN,
        make_encoder,
    )
except ImportError as e:            # Task 2 models are optional for Tasks 3-4
    _missing = e

    def __getattr__(name):
        raise ImportError(
            f"{name} lives in GTZAN_gnn.py, which could not be imported: "
            f"{_missing}") from _missing

__all__ = [
    "GraphSAGE",
    "GraphSAGEEncoder", "GATEncoder",
    "SingleBranchGNN", "TwoBranchFusionGNN", "make_encoder",
]
