# Reproducing the results

Everything below was run on an Apple M-series laptop (macOS, MPS backend,
Python 3.10). Nothing requires a GPU cluster, but Task 4 downloads 9.8 GB and
the full pipeline writes roughly 15 GB of intermediate data.

**Every stage is resumable.** If a stage's `.parquet` or cached `.pt` file
already exists it is reused and the stage is skipped. To force a rebuild, delete
the file. This matters: feature extraction on MagnaTagATune takes over an hour,
and a crash three stages later should not cost that.

---

## 0. Environment

```bash
git clone <repo-url> && cd gnn-bert-music-context
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch-geometric` sometimes needs its companion wheels installed separately; if
`import torch_geometric` fails, follow the install instructions for your torch
version at <https://pytorch-geometric.readthedocs.io>.

No script takes command-line arguments beyond the dispatchers. Each has a
`CONFIGURATION` block of named constants at the top of the file. `config.yaml`
records the values used for the reported results but does not drive behaviour.

---

## 1. Task 1 — BERT tag classification (MusicCaps captions)

No audio required. Both datasets download from the Hugging Face Hub.

```bash
python src/train.py --task 1
```

| | |
|---|---|
| Downloads | `humairaneha/MusicCaps-Curated-Tags` (~5 MB) |
| Runtime | ~20 min, 26 epochs |
| Writes | `results/task1_bert/` |

Expected: **Macro-F1 0.6016, Micro-F1 0.6394, AUC-PR 0.6462** over 65 tags on
504 test clips.

---

## 2. Task 2 — GNN on music structure graphs (GTZAN)

```bash
python src/audio_features.py --dataset gtzan     # ~15 min
python src/graph_builder.py  --dataset gtzan     # ~1 min
python src/train.py --task 2                     # ~30 min for all configs
```

| | |
|---|---|
| Downloads | `sanchit-gandhi/gtzan` (~1.2 GB) |
| Writes | `data/processed/gtzan/`, `results/task2_gnn/` |
| Disk | ~3 GB |

Expected validation Macro-F1: chroma-only **0.4682**, MFCC-only **0.7363**,
concat **0.6641**, two-branch **0.6880**.

To cut runtime, reduce `CONFIGS_TO_RUN` in `GTZAN_gnn.py` to
`["mfcc_sage"]` and set `RUN_CNN = False`.

---

## 3. Task 3 — Multi-label tagging and fusion (MagnaTagATune)

```bash
# check the tag partition before committing an hour to extraction
python -c "from MTAT_features import inspect_tags; inspect_tags()"

python src/audio_features.py --dataset mtat      # ~60-90 min
python src/graph_builder.py  --dataset mtat      # ~5 min
python src/train.py --task 3                     # ~20 min  GNN
python src/train.py --task 3 --variant mlp       # ~20 min  no-graph control
python src/train.py --task 3 --variant fusion    # ~60 min  five BERT ablations
```

| | |
|---|---|
| Downloads | `confit/magnatagatune` — `mp3.zip` 2.97 GB plus two CSVs |
| Writes | `data/processed/mtat/`, `results/task3_*/` |
| Disk | ~6 GB (3 GB audio, 1.9 GB frame features) |

Set `NUM_PROC = 4` in `MTAT_features.py` to parallelise extraction. If workers
hang, drop back to 1 — librosa with audioread under `fork` is unreliable on
macOS.

Expected test AUC-PR: GNN $\tau$=0.3 **0.2367**, GNN top-$k$=2 **0.2387**,
MLP no-graph **0.2464**.

### The edge-policy ablation

The default is `EDGE_POLICY = "tau"`. For the second arm, set
`EDGE_POLICY = "topk"` in `MTAT_graphs.py`, rerun `graph_builder.py`, then set
the same value in `gnn.py` and rerun. Graphs are cached per policy
(`train_mfcc_tau.pt`, `train_mfcc_topk.pt`), so the two builds coexist.

---

## 4. Task 4 — Cross-modal alignment (MusicCaps)

```bash
# confirm the audio mirror's columns before downloading 9.8 GB
python -c "from musiccaps_features import inspect_source; inspect_source()"

python src/audio_features.py --dataset musiccaps          # ~40 min
python src/graph_builder.py  --dataset musiccaps          # ~2 min
python src/train.py --task 4 --variant supervised         # ~10 min
python src/train.py --task 4                              # ~25 min
```

| | |
|---|---|
| Downloads | `CLAPv2/MusicCaps` (9.83 GB) — a community mirror of the audio |
| Writes | `data/processed/musiccaps/`, `results/task4_musiccaps/`, `results/musiccaps_supervised/` |
| Disk | ~11 GB |

Run the supervised variant **first**: the zero-shot section compares against its
`test_metrics.json` and silently omits that line if it is absent.

Expected: caption$\rightarrow$audio **R@1 0.0126, R@5 0.0766, R@10 0.1382**;
zero-shot tagging **AUC-PR 0.0535** against supervised **0.1502**.

The first Task 4 run encodes ~5,300 captions through frozen BERT and prints
progress. It takes 1–5 minutes and is cached; **do not interrupt it**, as the
cache is only written after the loop completes.

---

## 5. Aggregate

```bash
python src/evaluate.py
```

Walks `results/`, collects every `test_metrics.json` into
`results/metrics.json`, and prints the comparison tables used in the report.

---

## What will and will not reproduce exactly

**Deterministic.** Splits, tag partitions, graph construction and the derived
clip identifiers are all seeded or hash-based, so
`data/processed/` is byte-reproducible.

**Not exactly deterministic.** Training. MPS and CUDA kernels are not
bit-reproducible across machines, and we observed a run-to-run spread of
**±0.004 AUC-PR** on MagnaTagATune from two seeds of the same configuration.
Differences smaller than that in the reported tables are not resolvable, which
is why the edge-policy comparison is reported as a null result rather than a
ranking.

**Dataset drift.** `CLAPv2/MusicCaps` is a community mirror without a dataset
card. It contained 5,352 of MusicCaps' 5,521 clips when we downloaded it; a
different snapshot will change clip counts slightly, and the tag frequency floor
is applied *after* the join precisely so the label space always matches the
clips actually present.

**Two known data quirks**, both handled in code but worth knowing:
the audio mirror has no `ytid` column, so the join falls back to caption text
(valid, since MusicCaps captions are unique per clip); and the curated tag CSV's
`ytid` field is partly corrupted — IDs beginning with `-` were mangled to
`#NAME?` by a spreadsheet round-trip.

---

## Fastest path to a smoke test

To verify the pipeline end to end in under ten minutes without any large
download, set `MAX_CLIPS = 200` in `musiccaps_features.py` and run the Task 4
sequence. It exercises every stage — join, segmentation, feature extraction,
pooling, stratified split, graph construction, BERT caching, contrastive
training, retrieval and zero-shot evaluation — on a fraction of the data.
Restore `MAX_CLIPS = None` for the reported numbers.
