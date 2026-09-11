# Source modules

Run commands from the repository root. Dispatchers select the dataset-specific
implementation; model interface modules expose reusable classes and functions.

| Implementation | Entry point | Purpose |
|---|---|---|
| `bert_musiccaps_task1.py` | `train.py --task 1` | MusicCaps caption tagging |
| `GTZAN_features.py` | `audio_features.py --dataset gtzan` | Validate prepared splits and write label metadata |
| `GTZAN_graphs.py` | `graph_builder.py --dataset gtzan` | Build GTZAN graphs |
| `GTZAN_gnn.py` | `train.py --task 2` | Train GTZAN genre classifier |
| `MTAT_features.py` | `audio_features.py --dataset mtat` | Prepare MagnaTagATune features and splits |
| `MTAT_graphs.py` | `graph_builder.py --dataset mtat` | Build MagnaTagATune graphs |
| `gnn.py` | `train.py --task 3` | Train audio-only tag classifier |
| `task3_mlp_nograph.py` | `train.py --task 3 --variant mlp` | Train no-graph audio control |
| `task3_fusion.py` | `train.py --task 3 --variant fusion` | Train fusion and text controls |
| `musiccaps_features.py` | `audio_features.py --dataset musiccaps` | Prepare paired MusicCaps features and text |
| `run_musiccaps_supervised.py` | `train.py --task 4 --variant supervised` | Train supervised MusicCaps audio reference |
| `run_musiccaps_task4.py` | `train.py --task 4` | Configure MusicCaps contrastive training |
| `task4_contrastive.py` | Imported by `run_musiccaps_task4.py` | Dual encoder, retrieval, and zero-shot scoring |
| `evaluate.py` | `python src/evaluate.py` | Aggregate saved metrics |

`bert_encoder.py`, `gnn_model.py`, `fusion_model.py`, and `contrastive.py`
provide importable interfaces. Configuration constants live in the implementation
modules; `config.yaml` does not control execution.

The MusicCaps graph dispatcher currently targets `graphs.py`, which is absent
from this checkout. Existing MusicCaps graph caches can still be used by
training scripts. Graph rebuilding through that dispatcher requires its target
implementation.

Names and paths are case-sensitive on Linux. GTZAN uses
`data/processed/GTZAN/`; the other dataset directories are lowercase.
