# Where each file goes

The eight files with generic names are the interface the project specification
asks for. They are thin: dispatchers that launch a script, or re-exports that
expose a class. Every implementation lives in exactly one place, because two
copies of a model or a metric drift apart and then the ablation rows stop being
comparable.

## Drop your task scripts into this directory

| Your file | Reached through | Task |
|---|---|---|
| `GTZAN_features.py`  | `audio_features.py --dataset gtzan`      | 2 |
| `GTZAN_graphs.py`    | `graph_builder.py --dataset gtzan`       | 2 |
| `GTZAN_gnn.py`       | `train.py --task 2`, `gnn_model.py`      | 2 |
| `MTAT_features.py`   | `audio_features.py --dataset mtat`       | 3 |
| `MTAT_graphs.py`     | `graph_builder.py --dataset mtat`        | 3 |
| `gnn.py`             | `train.py --task 3`, `gnn_model.py`      | 3 |
| `task3_mlp_nograph.py` | `train.py --task 3 --variant mlp`      | 3 |
| `task3_fusion.py`    | `train.py --task 3 --variant fusion`, `fusion_model.py`, `bert_encoder.py` | 3 |
| `musiccaps_features.py` | `audio_features.py --dataset musiccaps` | 4 |
| `graphs.py`          | `graph_builder.py --dataset musiccaps`   | 4 |
| `task4_contrastive.py` | `contrastive.py`                       | 4 |
| `run_musiccaps_supervised.py` | `train.py --task 4 --variant supervised` | 4 |
| `run_musiccaps_task4.py` | `train.py --task 4`, `contrastive.py`  | 4 |

Task 1 (BERT on MusicCaps captions) is standalone and has no graph stage.

## Two things to check before committing

**Filenames are matched exactly, including case.** macOS is case-insensitive by
default, so `mtat_features.py` vs `MTAT_features.py` works locally and fails the
moment someone clones the repository on Linux. If a dispatcher cannot find its
target it prints the files it can actually see.

**`GTZAN_features.py` may not exist yet** under that name. Either rename your
GTZAN preprocessing script to match, or edit the one line in `TARGETS` at the
top of `audio_features.py`.

## Verify

```bash
cd src
python -c "import gnn_model, fusion_model, bert_encoder, contrastive; print('ok')"
python audio_features.py --help
python graph_builder.py --help
python train.py --help
```
