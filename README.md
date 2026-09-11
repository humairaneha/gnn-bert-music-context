# GNN-BERT Music Context Understanding

CSE715 supervised neural network project covering four tasks across MusicCaps,
GTZAN, and MagnaTagATune: BERT tagging, audio graph classification, GNN–BERT
fusion, and contrastive audio–text retrieval.

## Saved results

The following values come from the saved per-run metrics. Task 3 F1 values use
thresholds selected on validation and applied to test.

| Task | Dataset and model | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---:|---:|---:|
| 1 | MusicCaps BERT, 65 tags | 0.6016 | 0.6394 | 0.6462 |
| 2 | GTZAN MFCC GraphSAGE | 0.6024 | 0.6333 | 0.7164 |
| 3 | MagnaTagATune BERT-only | 0.2495 | 0.3787 | 0.2123 |
| 3 | MagnaTagATune GNN, tau | 0.2620 | 0.4480 | 0.2350 |
| 3 | MagnaTagATune GNN, top-k | 0.2586 | 0.4465 | 0.2368 |
| 3 | MagnaTagATune concatenation | 0.3074 | 0.5145 | 0.2945 |
| 3 | MagnaTagATune cross-attention | 0.3186 | 0.5161 | 0.3001 |
| 3 | MagnaTagATune MLP, no graph | 0.2605 | 0.4591 | 0.2464 |
| 3 | MagnaTagATune MLP + BERT | 0.3200 | 0.5296 | 0.3013 |

On the saved MagnaTagATune runs, the no-graph MLP exceeds both audio-only GNN
variants in AUC-PR. The GNN variants have higher genre F1 but lower mood F1 than
the MLP; their genre AUC-PR is also lower. Cross-attention improves over the
single-modality variants, while MLP + BERT achieves a slightly higher AUC-PR.
These saved comparisons do not establish statistical significance or a
consistent advantage from graph message passing.

Task 4 uses 796 MusicCaps test clips:

| Retrieval direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Caption → audio | 0.0126 | 0.0766 | 0.1382 |
| Audio → caption | 0.0126 | 0.0678 | 0.1332 |

Zero-shot tagging AUC-PR is **0.0535**, compared with **0.1502** for the saved
MusicCaps supervised audio-only GNN reference. This reference is distinct from
the MagnaTagATune Task 3 fusion model. Results across different datasets and
label spaces are not directly comparable.

Sources: `results/task1_bert/metrics.json`,
`results/task2/mfcc_sage_tau/test_metrics.json`, the per-variant
`results/task3_*/**/test_metrics.json` files,
`results/task4_musiccaps/test_metrics.json`, and
`results/musiccaps_supervised/mfcc_tau/test_metrics.json`.


## Dataset Fils

Due to repository size limitations, the `data/` directory is not included in this repository. 
The directory contains the raw data, processed features, graph representations, and train/validation/test splits required to reproduce the experiments.

To reproduce the results, download the dataset directory from the following link:

**Dataset download link:**  
https://drive.google.com/drive/folders/16CbINoBLB5tHmHbxkG8fRut24C0K9Ppj?usp=sharing

After downloading, extract the contents and place the `data/` directory directly inside the repository root so that the structure becomes:

```text
gnn-bert-music-context/
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── GTZAN/
│   │   ├── mtat/
│   │   └── musiccaps/
│   └── splits/
├── src/
├── results/
├── notebooks/
└── report/
```

## Layout

```text
src/
  audio_features.py       dispatcher --dataset {gtzan,mtat,musiccaps}
  graph_builder.py        dispatcher --dataset {gtzan,mtat,musiccaps}
  bert_encoder.py         frozen BERT cache interface for fusion
  gnn_model.py            GraphSAGE / GAT interfaces
  fusion_model.py         GNN–BERT fusion interface
  contrastive.py          Task 4 interface
  train.py                --task {1,2,3,4} [--variant ...]
  evaluate.py             aggregates saved metrics into results/metrics.json

  bert_musiccaps_task1.py                             Task 1
  GTZAN_features.py  GTZAN_graphs.py  GTZAN_gnn.py      Task 2
  MTAT_features.py   MTAT_graphs.py   gnn.py            Task 3
  task3_mlp_nograph.py  task3_fusion.py                 Task 3 controls/fusion
  musiccaps_features.py                               MusicCaps preprocessing
  run_musiccaps_supervised.py                         MusicCaps reference
  run_musiccaps_task4.py  task4_contrastive.py          Task 4

data/
  raw/
  processed/GTZAN/    processed/mtat/    processed/musiccaps/
  splits/
results/
  task1_bert/    task2/    task3_gnn_only/    task3_mlp_nograph/
  task3_fusion/  task4_musiccaps/  musiccaps_supervised/
  metrics.json  plots/    retrieval_examples/
notebooks/
  eda.ipynb
  demo_context.ipynb
report/
  figures/
```

Generic model interfaces and dispatchers refer to dataset-specific
implementations. `report/final_report.pdf` is not present in this checkout.
The task-by-task artifact mapping is in
[DELIVERABLES_FINDINGS_GUIDELINES.md](DELIVERABLES_FINDINGS_GUIDELINES.md).

## Running the pipeline

Run commands from the repository root after installing the dependencies:

```bash
pip install -r requirements.txt

# Task 1 — MusicCaps caption-to-tag classification
python src/train.py --task 1

# Task 2 — GTZAN, requires existing train/val/test parquet files
python src/audio_features.py --dataset gtzan
python src/graph_builder.py --dataset gtzan
python src/train.py --task 2

# Task 3 — MagnaTagATune
python src/audio_features.py --dataset mtat
python src/graph_builder.py --dataset mtat
python src/train.py --task 3
python src/train.py --task 3 --variant mlp
python src/train.py --task 3 --variant fusion

# Task 4 — MusicCaps
python src/audio_features.py --dataset musiccaps
# MusicCaps training requires existing graph caches; src/graphs.py is absent.
python src/train.py --task 4 --variant supervised
python src/train.py --task 4

# Aggregate saved results; this does not rerun model inference
python src/evaluate.py
```

The current `src/GTZAN_features.py` is a post-split label-space utility. It reads
`data/processed/GTZAN/train.parquet`, `val.parquet`, and `test.parquet`; it does
not extract features from raw audio. The GTZAN commands above therefore do not
constitute a complete raw-data reproduction pipeline. GTZAN results are written
to `results/task2/mfcc_sage_tau/` for the current configuration.

[REPRODUCE.md](REPRODUCE.md) describes environment setup, artifact dependencies,
and the available reproduction paths. Per-run metrics are the source for the
results above.

## Configuration

Dataset-specific scripts generally use constants in their configuration blocks.
`config.yaml` is a settings record, not a runtime configuration file, and some
values differ from the current saved runs: the GTZAN result records a hidden
dimension of 64, while the YAML model section lists 128.

The dispatchers accept task, variant, or dataset arguments as shown above.
`evaluate.py` also accepts `--verbose` to show the metric block selected for
each run. Feature and graph stages may reuse existing caches; cache reuse is
not equivalent to resuming a training checkpoint.


