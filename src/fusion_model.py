"""Public interfaces for GNN–BERT fusion, implemented in task3_fusion.py.

CrossAttention uses the graph readout as a query over text tokens and masks
padding before softmax. FusionModel supports:
    bert         BERT CLS features
    concat       GNN readout concatenated with BERT CLS
    crossattn    GNN readout with token-level cross-attention
    mlp_bert     mean-pooled audio MLP with BERT CLS
    tags_linear  classifier on instrument-tag vectors

The no-graph and raw-tag controls help distinguish the contributions of graph
message passing and BERT encoding from the input information itself.
"""

from task3_fusion import CrossAttention, FusionModel

__all__ = ["CrossAttention", "FusionModel"]
