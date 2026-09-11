"""
run_musiccaps_task4.py -- Task 4 contrastive alignment on MusicCaps.

Runs task4_contrastive.py against the MusicCaps graphs instead of MTAT. This is
the dataset the brief specifies for Task 4, and the reason matters: MusicCaps
captions are expert-written free text, unique per clip, so "which clip does this
caption describe" has exactly one answer and R@K measures what it should. MTAT's
templated instrument sentences repeat across hundreds of clips, which caps R@1
below 1 no matter how good the model is.

Overrides paths on the imported modules rather than editing them, so the MTAT
Task 3 runs stay reproducible.

Run run_musiccaps_supervised.py FIRST -- the zero-shot section compares against
its output, and will just skip that line if it is missing.

    python src/run_musiccaps_task4.py
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