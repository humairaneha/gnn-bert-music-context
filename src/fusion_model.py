"""
fusion_model.py -- GNN-BERT fusion, re-exported under the specification's name.

    CrossAttention   single-query cross-attention. The graph readout g is ONE
                     query attending over the L text tokens:
                         A = softmax(g W_Q (H W_K)^T / sqrt(d))
                         z = CONCAT(g, A H W_V)
                     Padded positions are masked to -inf before the softmax.

    FusionModel      one class, five ablation modes:
                       bert         BERT CLS only, no audio
                       concat       GNN + BERT CLS, early concatenation
                       crossattn    GNN + BERT tokens, cross-attention
                       mlp_bert     mean-pooled MLP + BERT CLS  (isolates the GRAPH)
                       tags_linear  linear probe on raw tags    (isolates BERT)

The last two are not in the specification but are what make the first three
interpretable: beating BERT-only shows that AUDIO helps, not that the GRAPH does.

Implementation lives in task3_fusion.py.
"""

from task3_fusion import CrossAttention, FusionModel

__all__ = ["CrossAttention", "FusionModel"]
