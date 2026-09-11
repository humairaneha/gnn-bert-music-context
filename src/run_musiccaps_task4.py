"""Run contrastive audio–text alignment on MusicCaps.

Configures task4_contrastive.py and its shared utilities with MusicCaps graph,
label, cache, and result paths. Retrieval uses the held-out audio–caption pairs.
Run the supervised MusicCaps variant first to make its tagging metrics
available for the zero-shot comparison.

    python src/train.py --task 4
"""

from pathlib import Path

import task3_fusion as t3f
import task4_contrastive as t4

MUSICCAPS = Path("data/processed/musiccaps")

# --- point the module at MusicCaps -------------------------------------------
t4.DATA_DIR = MUSICCAPS / "graphs"
t4.LABEL_SPACE = MUSICCAPS / "label_space.json"
t4.RESULT_DIR = Path("results/task4_musiccaps")
t4.FEATURE = "mfcc"
t4.EDGE_POLICY = "tau"

# The BERT cache path is a DEFAULT ARGUMENT of build_text_cache, bound when that
# function was defined, so reassigning task3_fusion.CACHE_DIR would not reach it
# -- and the MusicCaps captions would overwrite the MTAT cache in place. Wrap the
# call instead so each dataset keeps its own cache.
# max_len has the same problem, and it is not cosmetic: MusicCaps captions run
# well past 32 tokens, so the MTAT default silently truncates away most of the
# description the contrastive objective needs. Both are passed explicitly.
MAX_LEN = 256
_encode = t3f.build_text_cache
t4.build_text_cache = lambda graph_sets, **kw: _encode(
    graph_sets, max_len=MAX_LEN, cache_dir=MUSICCAPS / "bert_cache", **kw)
t4.CONTRASTIVE_BATCH = 128       # in-batch negatives are the entire signal
t4.EPOCHS = 40

# Captions are unique per clip here, so both of these become no-ops -- left on
# because they cost nothing and the code reports whether they fired.
t4.MASK_DUPLICATE_CAPTIONS = True
t4.GROUP_AWARE_RETRIEVAL = True

if __name__ == "__main__":
    print(f"Task 4 contrastive on MusicCaps -> {t4.RESULT_DIR}\n")
    t4.main()