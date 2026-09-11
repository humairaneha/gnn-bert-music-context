# Running and reproducing experiments

Run commands from the repository root. Saved per-run metrics describe the
reported experiments; current script constants may differ from those runs.
The result table in `README.md` identifies the retained scores.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The saved runs used macOS with an Apple M-series GPU and the MPS backend.
Training results may vary across devices and library versions. This checkout
does not establish a repeated-seed confidence interval.

## Required artifacts

| Operation | Inputs |
|---|---|
| Task 1 training | MusicCaps caption/tag dataset and pretrained BERT downloads |
| Graph construction | Dataset `train.parquet`, `val.parquet`, `test.parquet`, and label metadata |
| Graph-based training | Cached train/validation/test `.pt` graphs and label metadata |
| Demo inference | Test graph cache, `label_space.json`, model checkpoint, and thresholds |
| Metric aggregation | Saved per-run metric JSON files |

Raw audio, processed data, and checkpoints are generally excluded from Git.
A source-only clone therefore does not contain every artifact needed by the
demo or cached-graph training. Individual graph samples are not a substitute
for the full split caches.

## Task 1 — MusicCaps caption tagging

```bash
python src/train.py --task 1
```

The implementation is `src/bert_musiccaps_task1.py`. It fine-tunes BERT and
writes outputs under `results/task1_bert/`, including `metrics.json`, label
vocabulary, thresholds, split IDs, curves, and model files.

## Task 2 — GTZAN genre classification

The current feature utility requires prepared parquet splits under
`data/processed/GTZAN/`. It validates labels and split IDs; it does not extract
raw-audio features. A complete raw-to-parquet GTZAN pipeline is not provided by
this entry point.

```bash
python src/audio_features.py --dataset gtzan
python src/graph_builder.py --dataset gtzan
python src/train.py --task 2
```

Graphs are written under `data/processed/GTZAN/graphs/`; the default saved run
is under `results/task2/mfcc_sage_tau/`. Configuration constants in
`GTZAN_graphs.py` and `GTZAN_gnn.py` must use the same edge policy.

The CNN comparison contains a stored segment-level validation reference.
The referenced `cnn_eval.ipynb` is absent. GNN metrics are evaluated per track,
so the stored comparison does not use a common evaluation unit.

## Task 3 — MagnaTagATune tagging and fusion

```bash
python src/audio_features.py --dataset mtat
python src/graph_builder.py --dataset mtat
python src/train.py --task 3
python src/train.py --task 3 --variant mlp
python src/train.py --task 3 --variant fusion
```

Preprocessing reads the `confit/magnatagatune` audio and annotations. The
instrument-tag descriptions are inputs; genre and mood tags are targets.
Outputs are stored under `data/processed/mtat/` and `results/task3_*/`.

To inspect the tag partition without extracting audio, run from the repository
root:

```bash
PYTHONPATH=src python -c "from MTAT_features import inspect_tags; inspect_tags()"
```

Graph edge policies are configured in `MTAT_graphs.py`. Training modules must
select the corresponding cache. Some loaders fall back to an older cache name
when the policy-specific file is absent; confirm the loaded path when comparing
edge policies. The split is custom and song-grouped by default, which does not
establish artist separation.

## Task 4 — MusicCaps retrieval

```bash
PYTHONPATH=src python -c "from musiccaps_features import inspect_source; inspect_source()"
python src/audio_features.py --dataset musiccaps
```

The audio source is `CLAPv2/MusicCaps`; curated targets come from
`humairaneha/MusicCaps-Curated-Tags`. The audio download is approximately 9.8 GB.
The join supports caption matching when video IDs are unavailable. Dataset
availability and retained clip counts can change with the source snapshot.

The graph dispatcher targets `src/graphs.py`, which is absent from this
checkout. The following training commands require existing MusicCaps graph
caches under `data/processed/musiccaps/graphs/`:

```bash
python src/train.py --task 4 --variant supervised
python src/train.py --task 4
```

The supervised run provides an audio-only tagging reference on MusicCaps.
It is separate from the MagnaTagATune fusion model. Contrastive outputs are
stored under `results/task4_musiccaps/`; supervised outputs are under
`results/musiccaps_supervised/`. Frozen BERT states are cached during the run.

## Cache dependencies

Extraction produces `segmented_audio_data.parquet`; pooling produces
`segmented_audio_data_pooled.parquet`; splitting produces `train.parquet`,
`val.parquet`, and `test.parquet`. Completed extraction and pooling outputs are
reused. Removing an intermediate file requires rebuilding that stage when
preprocessing is run again.

Existing graph-based training and inference do not require the intermediate
parquet files. Graph rebuilding and parquet-based analysis require their
respective split files. Cache reuse does not imply that model training resumes
from a checkpoint.

## Aggregate results

```bash
python src/evaluate.py --verbose
```

This collects saved metric files into `results/metrics.json`; it does not run
model inference. The verbose output identifies the selected metric block.
Validation-tuned and fixed-0.5 results are stored separately in multi-label
runs. Retain per-run source files when comparing experiments.

## Demo notebook

Open `notebooks/demo_context.ipynb` with the working directory set to
`notebooks/`. It loads a cached MagnaTagATune test graph and runs the audio-only
GNN. Its inputs include the test graph cache, label metadata, `best_gnn.pt`,
and `thresholds.npy`. The displayed description is metadata; the notebook does
not currently perform BERT fusion inference.
