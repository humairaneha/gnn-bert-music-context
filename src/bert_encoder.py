"""
bert_encoder.py -- frozen BERT text encoding, re-exported under the
specification's name.

    build_text_cache(graph_sets, model_name, max_len, cache_dir)
        Encodes every UNIQUE text string once and returns
            H[str]      -> [max_len, 768]  token-level states
            mask[str]   -> [max_len]       attention mask
            tokens[str] -> list[str]       for attention visualisation

BERT is frozen throughout. Fine-tuning 110M parameters against a few thousand
captions would memorise rather than generalise, and would break parameter parity
across the fusion ablation rows. Token-level states are kept, not just CLS,
because cross-attention needs something to attend over.

The cache is keyed on the text set, so it invalidates itself if the captions or
the tag partition change.

Implementation lives in task3_fusion.py.
"""

from task3_fusion import build_text_cache, lookup_text

__all__ = ["build_text_cache", "lookup_text"]


# Task 1 fine-tunes BERT with a classification head; Tasks 3 and 4 use it frozen.
# Both live here so "the BERT encoder" is one file, as the specification implies.
try:
    from bert_musiccaps_task1 import build_model as build_task1_classifier
    __all__.append("build_task1_classifier")
except ImportError:
    pass
