"""Public interfaces for BERT text encoding.

build_text_cache() and lookup_text() are implemented in task3_fusion.py.
The cache stores frozen token-level states, attention masks, and tokens for
unique input strings. Token states support cross-attention; CLS states support
text-only and concatenation models.

build_task1_classifier is available when Task 1 dependencies can be imported.
Task 1 fine-tunes BERT; the fusion and retrieval pipelines keep it frozen.
"""

from task3_fusion import build_text_cache, lookup_text

__all__ = ["build_text_cache", "lookup_text"]


# Task 1 fine-tunes BERT with a classification head; Tasks 3 and 4 use it frozen.
# Expose the optional Task 1 classifier alongside the frozen-cache utilities.
try:
    from bert_musiccaps_task1 import build_model as build_task1_classifier
    __all__.append("build_task1_classifier")
except ImportError:
    pass
