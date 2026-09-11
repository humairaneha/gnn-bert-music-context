# Task-by-task deliverables mapping

Project guide: `CSE425_Project_GNN_BERT_Music_Context.pdf`.
Repository: `gnn-bert-music-context/`.
All repository paths below are relative to the repository root.

## Task 1 — BERT Baseline for Music Tag Understanding

Project guide: Section 4.1, page 3.

| Project-guide deliverable | Repository mapping |
|---|---|
| BERT fine-tuning code | `src/bert_musiccaps_task1.py`; entry point: `src/train.py --task 1` |
| MusicCaps caption-to-tag proxy results | `results/task1_bert/metrics.json` |
| Macro-F1 and Micro-F1 curves versus epochs | `results/task1_bert/f1_curves.png`; `results/plots/task1_bert/f1_curves.png` |
| Five example predictions; optional attention visualization | No dedicated saved five-prediction artifact identified in `results/task1_bert/` |
| Supporting per-tag and category results | `results/task1_bert/per_tag_test.csv`; `results/task1_bert/per_category_test.csv` |
| Supporting model and vocabulary | `results/task1_bert/final/`; `results/task1_bert/vocabulary.json`; `results/task1_bert/vocabulary_selection.csv` |
| Supporting split identifiers | `results/task1_bert/split_train.json`; `split_validation.json`; `split_test.json` |

Saved results: 65 tags, 504 test clips, Macro-F1 **0.6016**, Micro-F1 **0.6394**, AUC-PR **0.6462**.

## Task 2 — GNN on Music Structure Graphs

Project guide: Section 4.2, page 4.

| Project-guide deliverable | Repository mapping |
|---|---|
| Chroma/MFCC graph construction scripts | `src/GTZAN_graphs.py`; dispatcher: `src/graph_builder.py` |
| PyTorch Geometric GNN implementation | `src/GTZAN_gnn.py`; interface: `src/gnn_model.py` |
| Genre classification on GTZAN | `results/task2/mfcc_sage_tau/test_metrics.json`; `classification_report.json`; `classification_report.txt`; `per_class_metrics.csv`; `test_predictions.csv` |
| Comparison against mel-spectrogram CNN | `results/task2/mfcc_sage_tau/cnn_gnn_comparison.json`; `cnn_gnn_comparison.csv`; `cnn_gnn_comparison.png` |
| Supporting training curves | `results/task2/mfcc_sage_tau/loss_curve.png`; `f1_curve.png`; `training_history.json` |
| Supporting confusion matrix and embeddings plot | `results/task2/mfcc_sage_tau/confusion_matrix.png`; `tsne.png` |
| Supporting graph samples | `data/processed/GTZAN/graph_samples/` — 20 `.pt` samples |

Saved GNN test results: Macro-F1 **0.6024**, Micro-F1 **0.6333**, AUC-PR **0.7164**.

## Task 3 — GNN–BERT Fusion for Multi-Context Understanding

Project guide: Section 4.3, page 4.

| Project-guide deliverable | Repository mapping |
|---|---|
| GNN–BERT fusion model | `src/task3_fusion.py`; interface: `src/fusion_model.py`; text interface: `src/bert_encoder.py` |
| BERT-only ablation | `results/task3_fusion/bert/test_metrics.json` |
| GNN-only ablation | `src/gnn.py`; `results/task3_gnn_only/mfcc_tau/test_metrics.json` |
| Early-concatenation ablation | `results/task3_fusion/concat/test_metrics.json` |
| Cross-attention ablation | `results/task3_fusion/crossattn/test_metrics.json` |
| MagnaTagATune Macro-F1 and AUC-PR results | Per-variant `test_metrics.json`; `results/task3_fusion/ablation_metrics.json` |
| t-SNE colored by genre and mood | `results/plots/task3_fusion/bert/tsne_genre_mood.png`; `results/plots/task3_fusion/concat/tsne_genre_mood.png`; `results/plots/task3_fusion/crossattn/tsne_genre_mood.png` |
| Three graph-path and text-alignment case studies | `results/task3_fusion/case_studies.json` — 3 records containing graph edges, text, attended tokens, true tags, predictions, and errors |
| Additional no-graph controls | `src/task3_mlp_nograph.py`; `results/task3_mlp_nograph/mfcc_nograph/test_metrics.json`; `results/task3_fusion/mlp_bert/test_metrics.json` |
| Supporting graph samples | `data/processed/mtat/graph_samples/` — 29 `.pt` samples |

Saved MagnaTagATune results, using validation-tuned thresholds:

| Model | Macro-F1 | Micro-F1 | AUC-PR |
|---|---:|---:|---:|
| BERT-only | 0.2495 | 0.3787 | 0.2123 |
| GNN-only, tau | 0.2620 | 0.4480 | 0.2350 |
| Early concatenation | 0.3074 | 0.5145 | 0.2945 |
| Cross-attention | 0.3186 | 0.5161 | 0.3001 |

The fusion implementation uses frozen, cached BERT representations. The case-study text consists of instrument-tag-derived descriptions.

## Task 4 — Cross-Modal MusicCaps Alignment

Project guide: Section 4.4, page 5; human evaluation: Section 6, page 6.

| Project-guide deliverable | Repository mapping |
|---|---|
| GNN–BERT dual encoder with contrastive training | `src/run_musiccaps_task4.py`; `src/task4_contrastive.py`; interface: `src/contrastive.py` |
| Retrieval evaluation on MusicCaps test split | `results/task4_musiccaps/test_metrics.json`, fields `caption_to_audio` and `audio_to_caption` |
| Ten caption queries with top-three matched clips | `results/task4_musiccaps/qualitative_retrieval.json` — 10 queries, each with 3 retrieved clips; copy: `results/retrieval_examples/task4_musiccaps/qualitative_retrieval.json` |
| Zero-shot tag prediction | `results/task4_musiccaps/test_metrics.json`, field `zero_shot_tagging` |
| Comparison with a supervised model | `results/musiccaps_supervised/mfcc_tau/test_metrics.json`; implementation: `src/run_musiccaps_supervised.py`. This saved comparator is a MusicCaps audio-only GNN, not the Task 3 fusion model. |
| Human evaluation by at least five listeners on a 1–5 scale | `results/task4_contrastive/human_eval_sheet.csv`; copy: `results/retrieval_examples/task4_contrastive/human_eval_sheet.csv`. The inspected sheet has 20 rows and no completed rater scores. |
| Supporting model and shared embeddings | `results/task4_musiccaps/best_dual_encoder.pt`; `results/task4_musiccaps/shared_space.pt` |
| Supporting graph samples | `data/processed/musiccaps/graph_samples/` — 20 `.pt` samples |

Saved retrieval results on 796 test clips:

| Direction | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| Caption → audio | 0.0126 | 0.0766 | 0.1382 |
| Audio → caption | 0.0126 | 0.0678 | 0.1332 |

Saved zero-shot tagging AUC-PR: **0.0535**. Saved supervised MusicCaps GNN AUC-PR: **0.1502**.
