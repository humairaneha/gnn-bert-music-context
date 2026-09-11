"""
Task 1: BERT multi-label tag classifier for MusicCaps captions.

Loads MusicCaps-Curated-Tags, keeps the genre, mood and tempo tags that have
enough examples to be scored, splits the data without leakage, fine-tunes BERT,
and reports Macro-F1, Micro-F1 and AUC-PR using a decision threshold tuned per
tag on the validation set.

Early stopping monitors validation loss to reduce overfitting.

    python bert_musiccaps_task1.py
"""

import inspect
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# ============================================================
# Configuration
# ============================================================

DATASET = "humairaneha/MusicCaps-Curated-Tags"
BERT_MODEL = "bert-base-uncased"
SAVE_DIR = Path("musiccaps_bert_task1")

KEEP_CATEGORIES = ["genre", "mood", "tempo"]
MIN_EXAMPLES_PER_TAG = 45

MAX_TOKENS = 256
BATCH_SIZE = 32

# Large upper bound. Early stopping will usually stop earlier.
MAX_EPOCHS = 50

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-5
WARMUP_RATIO = 0.1

# Early stopping
# Stop after this many consecutive epochs without sufficient
# improvement in validation loss.
STOP_AFTER_BAD_EPOCHS = 4

# Minimum amount by which validation loss must improve.
EARLY_STOPPING_THRESHOLD = 0.001

BALANCE_RARE_TAGS = True
MAX_TAG_WEIGHT = 10.0
SEED = 42


# ============================================================
# Data utilities
# ============================================================

def clean_caption(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def load_data():
    """Load the wide one-hot table and the tag frequency table."""

    df = load_dataset(DATASET, split="train").to_pandas()

    tag_info = load_dataset(
        DATASET,
        "tag_statistics",
        split="train"
    ).to_pandas()

    tag_info.columns = [
        name.strip().lstrip("\ufeff")
        for name in tag_info.columns
    ]

    print(f"loaded {len(df)} clips and {len(tag_info)} tags")

    return df, tag_info


def choose_tags(df, tag_info):
    """
    Keep in-scope tags with at least MIN_EXAMPLES_PER_TAG positive clips.

    Rarer tags stay in the dataframe but are not predicted or scored.
    Below about fifty positives, a per-tag F1 on a ten percent test split
    is mostly noise.
    """

    tag_category = dict(
        zip(tag_info["tag"], tag_info["category"])
    )

    in_scope = tag_info[
        tag_info["category"].isin(KEEP_CATEGORIES)
    ]["tag"].tolist()

    candidates = [
        tag for tag in in_scope
        if tag in df.columns
    ]

    no_column = sorted(
        set(in_scope) - set(candidates)
    )

    if no_column:
        print(
            f"{len(no_column)} in-scope tags have no column: "
            f"{no_column[:8]}"
        )

    counts = df[candidates].sum().astype(int)

    kept_tags = sorted(
        counts[counts >= MIN_EXAMPLES_PER_TAG].index
    )

    summary = pd.DataFrame(
        {
            "tag": candidates,
            "category": [
                tag_category.get(tag, "unknown")
                for tag in candidates
            ],
            "examples": [
                int(counts[tag])
                for tag in candidates
            ],
        }
    )

    summary["kept"] = summary["tag"].isin(kept_tags)

    summary = summary.sort_values(
        ["category", "examples"],
        ascending=[True, False]
    )

    print(f"\ncategories in scope: {KEEP_CATEGORIES}")
    print(
        f"candidates: {len(candidates)} | "
        f"minimum examples: {MIN_EXAMPLES_PER_TAG}"
    )
    print(
        f"kept: {len(kept_tags)} | "
        f"dropped: {len(candidates) - len(kept_tags)}"
    )

    print("\nFINAL VOCABULARY")

    for category in KEEP_CATEGORIES:
        rows = summary[
            (summary["category"] == category)
            & summary["kept"]
        ]

        print(f"\n  {category} ({len(rows)})")

        for row in rows.itertuples():
            print(
                f"    {row.tag:<22s} {row.examples:5d}"
            )

    print("\nDROPPED (kept in the data, not scored)")

    for category in KEEP_CATEGORIES:

        rows = summary[
            (summary["category"] == category)
            & ~summary["kept"]
        ]

        if len(rows):

            listing = ", ".join(
                f"{row.tag} ({row.examples})"
                for row in rows.itertuples()
            )

            print(
                f"  {category} ({len(rows)}): {listing}"
            )

    per_category = (
        summary[summary["kept"]]["category"]
        .value_counts()
        .to_dict()
    )

    print(
        f"\nlabel space: {len(kept_tags)} tags "
        f"-> {per_category}\n"
    )

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        SAVE_DIR / "vocabulary_selection.csv",
        index=False
    )

    return kept_tags, tag_category


def clean_data(df, tags):
    """
    Drop duplicate clips and clips left without any of the kept tags.
    """

    id_column = (
        "ytid"
        if "ytid" in df.columns
        else df.columns[0]
    )

    clips = df[
        [id_column, "caption"] + tags
    ].rename(
        columns={id_column: "clip_id"}
    )

    clips["caption_key"] = clips[
        "caption"
    ].map(clean_caption)

    before = len(clips)

    clips = clips[
        clips["caption_key"].str.len() > 0
    ]

    clips = clips.drop_duplicates(
        subset="clip_id",
        keep="first"
    )

    clips = clips.drop_duplicates(
        subset="caption_key",
        keep="first"
    )

    has_tag = clips[tags].sum(axis=1) > 0

    clips = clips[
        has_tag
    ].reset_index(drop=True)

    labels = clips[
        tags
    ].values.astype(np.float32)

    tags_per_clip = labels.sum(axis=1)

    print(
        f"clips: {before} -> {len(clips)} "
        f"after removing duplicates and untagged rows"
    )

    print(
        f"tags per clip: "
        f"mean {tags_per_clip.mean():.2f}, "
        f"median {np.median(tags_per_clip):.0f}"
    )

    return clips, labels


def split_data(clips, labels):
    """
    Split 80/10/10 with multilabel stratification,
    then check for leakage.
    """

    captions = clips["caption"].values

    first = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SEED
    )

    train_rows, holdout_rows = next(
        first.split(captions, labels)
    )

    second = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=SEED
    )

    val_part, test_part = next(
        second.split(
            captions[holdout_rows],
            labels[holdout_rows]
        )
    )

    splits = {
        "train": train_rows,
        "validation": holdout_rows[val_part],
        "test": holdout_rows[test_part],
    }

    check_no_leakage(
        clips,
        splits
    )

    for name, rows in splits.items():

        counts = labels[
            rows
        ].sum(axis=0)

        print(
            f"{name:11s} "
            f"{len(rows):5d} clips | "
            f"tags with no examples: "
            f"{int((counts == 0).sum()):3d} | "
            f"smallest tag: {int(counts.min())}"
        )

    return splits


def check_no_leakage(clips, splits):
    """
    Stop if the same clip or the same caption text
    lands in two splits.
    """

    for column, description in [
        ("clip_id", "clip id"),
        ("caption_key", "caption"),
    ]:

        where = {}

        for name, rows in splits.items():

            for value in clips.loc[
                rows,
                column
            ]:

                if (
                    value in where
                    and where[value] != name
                ):

                    raise SystemExit(
                        f"leakage: {description} "
                        f"{value!r} in "
                        f"{where[value]} and {name}"
                    )

                where[value] = name


def make_datasets(
    clips,
    labels,
    splits,
    tokenizer
):
    """
    Turn each split into a tokenised HuggingFace dataset.
    """

    def tokenize(batch):

        return tokenizer(
            batch["caption"],
            truncation=True,
            max_length=MAX_TOKENS
        )

    datasets = {}

    for name, rows in splits.items():

        table = pd.DataFrame(
            {
                "caption": clips.loc[
                    rows,
                    "caption"
                ].values,

                "labels": list(
                    labels[rows]
                ),
            }
        )

        dataset = Dataset.from_pandas(
            table,
            preserve_index=False
        )

        datasets[name] = dataset.map(
            tokenize,
            batched=True,
            remove_columns=["caption"]
        )

    return datasets


# ============================================================
# Training utilities
# ============================================================

def make_tag_weights(train_labels):
    """
    Weight each tag by how rare it is,
    capped so training stays stable.

    Without the cap a one-percent tag gets a weight near
    a hundred, which costs a lot of precision without
    buying much recall.
    """

    positives = train_labels.sum(axis=0)

    negatives = (
        len(train_labels) - positives
    )

    weights = (
        negatives
        / np.maximum(positives, 1.0)
    )

    return np.clip(
        weights,
        1.0,
        MAX_TAG_WEIGHT
    ).astype(np.float32)


class WeightedTrainer(Trainer):
    """
    Trainer that applies per-tag weights
    inside the BCE loss.
    """

    def __init__(
        self,
        tag_weights=None,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.tag_weights = tag_weights


    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        weights = None

        if self.tag_weights is not None:

            weights = self.tag_weights.to(
                outputs.logits.device
            )

        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=weights
        )

        loss = loss_fn(
            outputs.logits,
            labels.float()
        )

        if return_outputs:
            return loss, outputs

        return loss


def epoch_metrics(eval_pred):
    """
    Metrics reported each epoch.

    These metrics are only monitored for interpretation.
    Early stopping is based on validation loss.
    """

    logits, labels = eval_pred

    probs = 1 / (
        1 + np.exp(-logits)
    )

    preds = (
        probs >= 0.3
    ).astype(int)

    return {
        "micro_f1": f1_score(
            labels,
            preds,
            average="micro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            labels,
            preds,
            average="macro",
            zero_division=0
        ),

        "micro_precision": precision_score(
            labels,
            preds,
            average="micro",
            zero_division=0
        ),

        "micro_recall": recall_score(
            labels,
            preds,
            average="micro",
            zero_division=0
        ),
    }


# ============================================================
# Threshold tuning
# ============================================================

def find_best_thresholds(
    probs,
    labels,
    grid=np.arange(
        0.05,
        0.91,
        0.05
    )
):
    """
    Pick the threshold that maximises F1 for each tag,
    using validation only.

    Tags with no validation examples keep 0.5,
    since any tuned value would be guesswork.
    """

    thresholds = np.full(
        probs.shape[1],
        0.5
    )

    for tag in range(
        probs.shape[1]
    ):

        if labels[:, tag].sum() == 0:
            continue

        scores = [
            f1_score(
                labels[:, tag],
                probs[:, tag] >= t,
                zero_division=0
            )
            for t in grid
        ]

        thresholds[tag] = grid[
            int(np.argmax(scores))
        ]

    return thresholds


# ============================================================
# Evaluation
# ============================================================

def score_overall(
    probs,
    labels,
    thresholds,
    tags
):
    """
    Headline metrics on the test split.
    """

    preds = (
        probs >= thresholds
    ).astype(int)

    has_examples = (
        labels.sum(axis=0) > 0
    )

    auc_scores = [
        average_precision_score(
            labels[:, i],
            probs[:, i]
        )
        for i in range(len(tags))
        if labels[:, i].sum() > 0
    ]

    return {
        "n_tags": len(tags),

        "n_tags_with_test_examples": int(
            has_examples.sum()
        ),

        "test_clips": int(
            len(labels)
        ),

        "macro_f1": float(
            f1_score(
                labels,
                preds,
                average="macro",
                zero_division=0
            )
        ),

        "micro_f1": float(
            f1_score(
                labels,
                preds,
                average="micro",
                zero_division=0
            )
        ),

        "macro_f1_scored_tags": float(
            f1_score(
                labels[:, has_examples],
                preds[:, has_examples],
                average="macro",
                zero_division=0
            )
        ),

        "micro_precision": float(
            precision_score(
                labels,
                preds,
                average="micro",
                zero_division=0
            )
        ),

        "micro_recall": float(
            recall_score(
                labels,
                preds,
                average="micro",
                zero_division=0
            )
        ),

        "macro_auc_pr": float(
            np.mean(auc_scores)
        ),
    }


def score_each_tag(
    probs,
    labels,
    thresholds,
    tags,
    tag_category
):
    """
    Precision, recall, F1 and AUC-PR
    for every tag.
    """

    preds = (
        probs >= thresholds
    ).astype(int)

    rows = []

    for i, tag in enumerate(tags):

        n_examples = int(
            labels[:, i].sum()
        )

        rows.append(
            {
                "tag": tag,

                "category":
                    tag_category.get(
                        tag,
                        "unknown"
                    ),

                "test_examples":
                    n_examples,

                "threshold":
                    round(
                        float(
                            thresholds[i]
                        ),
                        2
                    ),

                "precision":
                    precision_score(
                        labels[:, i],
                        preds[:, i],
                        zero_division=0
                    ),

                "recall":
                    recall_score(
                        labels[:, i],
                        preds[:, i],
                        zero_division=0
                    ),

                "f1":
                    f1_score(
                        labels[:, i],
                        preds[:, i],
                        zero_division=0
                    ),

                "auc_pr":
                    (
                        average_precision_score(
                            labels[:, i],
                            probs[:, i]
                        )
                        if n_examples > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "test_examples",
        ascending=False
    )


# ============================================================
# Curves
# ============================================================

def save_curves(
    log_history,
    path
):
    """
    Save training loss, validation loss
    and validation F1 curves.

    The loss graph is particularly useful for
    checking whether overfitting occurred.
    """

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    log = pd.DataFrame(
        log_history
    )

    train_log = log.dropna(
        subset=["loss"]
    )

    eval_log = log.dropna(
        subset=["eval_loss"]
    )

    figure, (
        loss_plot,
        f1_plot
    ) = plt.subplots(
        1,
        2,
        figsize=(11, 4)
    )

    # ----------------------------
    # Loss curves
    # ----------------------------

    loss_plot.plot(
        train_log["epoch"],
        train_log["loss"],
        label="train"
    )

    loss_plot.plot(
        eval_log["epoch"],
        eval_log["eval_loss"],
        label="validation"
    )

    loss_plot.set_xlabel("epoch")
    loss_plot.set_ylabel("BCE loss")
    loss_plot.legend()

    # ----------------------------
    # F1 curves
    # ----------------------------

    f1_plot.plot(
        eval_log["epoch"],
        eval_log["eval_macro_f1"],
        marker="o",
        label="Macro-F1"
    )

    f1_plot.plot(
        eval_log["epoch"],
        eval_log["eval_micro_f1"],
        marker="o",
        label="Micro-F1"
    )

    f1_plot.set_xlabel("epoch")
    f1_plot.set_ylabel("validation F1")
    f1_plot.legend()

    figure.tight_layout()

    figure.savefig(
        path,
        dpi=150
    )

    plt.close(figure)


# ============================================================
# Example predictions
# ============================================================

def show_examples(
    captions,
    labels,
    probs,
    preds,
    tags,
    n=5
):
    """
    Print a few test captions with
    their true and predicted tags.
    """

    print(
        f"\nexample predictions ({n})"
    )

    picker = np.random.default_rng(0)

    for row in picker.choice(
        len(captions),
        size=min(
            n,
            len(captions)
        ),
        replace=False
    ):

        true_tags = [
            tags[i]
            for i in np.flatnonzero(
                labels[row]
            )
        ]

        predicted = [
            (
                tags[i],
                probs[row, i]
            )
            for i in np.flatnonzero(
                preds[row]
            )
        ]

        predicted.sort(
            key=lambda pair: -pair[1]
        )

        print(
            f"\ncaption: "
            f"{captions[row][:220]}"
        )

        print(
            "  true      : "
            + (
                ", ".join(true_tags)
                if true_tags
                else "-"
            )
        )

        print(
            "  predicted : "
            + (
                ", ".join(
                    f"{tag} ({p:.2f})"
                    for tag, p in predicted
                )
                if predicted
                else "-"
            )
        )


# ============================================================
# TrainingArguments compatibility
# ============================================================

def build_training_settings(
    **options
):
    """
    Create TrainingArguments while adjusting for the
    installed transformers version.

    Some transformers versions use evaluation_strategy
    instead of eval_strategy.

    Transformers 5 may also replace warmup_ratio
    with warmup_steps.
    """

    accepted = set(
        inspect.signature(
            TrainingArguments.__init__
        ).parameters
    )

    if (
        "warmup_ratio" in options
        and "warmup_ratio" not in accepted
    ):
        options["warmup_steps"] = (
            options.pop(
                "warmup_ratio"
            )
        )

    if (
        "eval_strategy" in options
        and "eval_strategy" not in accepted
    ):
        options[
            "evaluation_strategy"
        ] = options.pop(
            "eval_strategy"
        )

    return TrainingArguments(
        **options
    )


# ============================================================
# Main
# ============================================================

def main():

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df, tag_info = load_data()

    tags, tag_category = choose_tags(
        df,
        tag_info
    )

    clips, labels = clean_data(
        df,
        tags
    )

    splits = split_data(
        clips,
        labels
    )

    # --------------------------------------------------------
    # Tokenizer and datasets
    # --------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BERT_MODEL
        )
    )

    datasets = make_datasets(
        clips,
        labels,
        splits,
        tokenizer
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BERT_MODEL,
            num_labels=len(tags),

            id2label={
                i: tag
                for i, tag
                in enumerate(tags)
            },

            label2id={
                tag: i
                for i, tag
                in enumerate(tags)
            },

            problem_type=
                "multi_label_classification",
        )
    )

    # --------------------------------------------------------
    # Training settings
    # --------------------------------------------------------
    #
    # IMPORTANT
    #
    # Validation loss determines the best model.
    #
    # Early stopping therefore stops training when
    # validation loss has failed to improve sufficiently
    # for STOP_AFTER_BAD_EPOCHS epochs.
    #
    # This helps prevent continued fitting of the training
    # data after validation performance begins degrading.
    #
    # --------------------------------------------------------

    settings = build_training_settings(

        output_dir=str(
            SAVE_DIR / "checkpoints"
        ),

        learning_rate=LEARNING_RATE,

        warmup_ratio=WARMUP_RATIO,

        per_device_train_batch_size=
            BATCH_SIZE,

        per_device_eval_batch_size=
            BATCH_SIZE,

        num_train_epochs=
            MAX_EPOCHS,

        weight_decay=
            WEIGHT_DECAY,

        # Evaluate every epoch
        eval_strategy="epoch",

        # Save every epoch
        save_strategy="epoch",

        # Reload the epoch with the lowest
        # validation loss after training
        load_best_model_at_end=True,

        # -------------------------------
        # EARLY STOPPING TARGET
        # -------------------------------

        metric_for_best_model=
            "eval_loss",

        greater_is_better=False,

        # -------------------------------

        save_total_limit=2,

        logging_strategy="epoch",

        seed=SEED,

        report_to="none",
    )

    # --------------------------------------------------------
    # Optional rare-tag weighting
    # --------------------------------------------------------

    tag_weights = None

    if BALANCE_RARE_TAGS:

        tag_weights = torch.tensor(
            make_tag_weights(
                labels[
                    splits["train"]
                ]
            )
        )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = WeightedTrainer(

        tag_weights=tag_weights,

        model=model,

        args=settings,

        train_dataset=
            datasets["train"],

        eval_dataset=
            datasets["validation"],

        data_collator=
            DataCollatorWithPadding(
                tokenizer=tokenizer
            ),

        processing_class=
            tokenizer,

        compute_metrics=
            epoch_metrics,

        # ----------------------------------------------------
        # EARLY STOPPING
        #
        # Stop if validation loss fails to improve by at
        # least EARLY_STOPPING_THRESHOLD for
        # STOP_AFTER_BAD_EPOCHS consecutive evaluations.
        # ----------------------------------------------------

        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=
                    STOP_AFTER_BAD_EPOCHS,

                early_stopping_threshold=
                    EARLY_STOPPING_THRESHOLD,
            )
        ],
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_result = trainer.train()

    # Report when training actually stopped
    print("\nTRAINING COMPLETE")
    print(
        f"epochs completed: "
        f"{trainer.state.epoch:.2f}"
    )

    print(
        f"best checkpoint: "
        f"{trainer.state.best_model_checkpoint}"
    )

    print(
        f"best validation loss: "
        f"{trainer.state.best_metric}"
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    val_output = trainer.predict(
        datasets["validation"]
    )

    val_probs = 1 / (
        1
        + np.exp(
            -val_output.predictions
        )
    )

    thresholds = find_best_thresholds(
        val_probs,
        val_output.label_ids
    )

    # --------------------------------------------------------
    # Test predictions
    # --------------------------------------------------------

    test_output = trainer.predict(
        datasets["test"]
    )

    test_probs = 1 / (
        1
        + np.exp(
            -test_output.predictions
        )
    )

    test_labels = (
        test_output.label_ids
    )

    test_preds = (
        test_probs >= thresholds
    ).astype(int)

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    results = score_overall(
        test_probs,
        test_labels,
        thresholds,
        tags
    )

    # Add training information
    results[
        "epochs_completed"
    ] = float(
        trainer.state.epoch
    )

    results[
        "best_validation_loss"
    ] = float(
        trainer.state.best_metric
    )

    results[
        "best_checkpoint"
    ] = str(
        trainer.state.best_model_checkpoint
    )

    print("\ntest metrics")

    print(
        json.dumps(
            results,
            indent=2
        )
    )

    # --------------------------------------------------------
    # Per-tag metrics
    # --------------------------------------------------------

    tag_scores = score_each_tag(
        test_probs,
        test_labels,
        thresholds,
        tags,
        tag_category
    )

    scored = tag_scores[
        tag_scores["test_examples"] > 0
    ]

    category_scores = (
        scored.groupby("category")
        .agg(
            tags=("tag", "size"),
            macro_f1=("f1", "mean"),
            macro_auc_pr=(
                "auc_pr",
                "mean"
            ),
        )
        .round(4)
    )

    print("\nper category")

    print(
        category_scores.to_string()
    )

    print("\nweakest tags")

    print(
        scored.nsmallest(
            10,
            "f1"
        )[
            [
                "tag",
                "category",
                "test_examples",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    show_examples(

        clips.loc[
            splits["test"],
            "caption"
        ].values,

        test_labels,

        test_probs,

        test_preds,

        tags
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    trainer.save_model(
        SAVE_DIR / "final"
    )

    tokenizer.save_pretrained(
        SAVE_DIR / "final"
    )

    # --------------------------------------------------------
    # Save thresholds
    # --------------------------------------------------------

    np.save(
        SAVE_DIR / "thresholds.npy",
        thresholds
    )

    (
        SAVE_DIR
        / "thresholds.json"
    ).write_text(
        json.dumps(
            dict(
                zip(
                    tags,
                    thresholds
                    .round(2)
                    .tolist()
                )
            ),
            indent=2
        )
    )

    # --------------------------------------------------------
    # Save vocabulary metadata
    # --------------------------------------------------------

    (
        SAVE_DIR
        / "vocabulary.json"
    ).write_text(
        json.dumps(
            {
                "tags": tags,

                "n_tags":
                    len(tags),

                "categories": {
                    tag:
                        tag_category.get(
                            tag,
                            "unknown"
                        )
                    for tag in tags
                },

                "examples": {
                    tag:
                        int(
                            labels[:, i]
                            .sum()
                        )
                    for i, tag
                    in enumerate(tags)
                },

                "min_examples_per_tag":
                    MIN_EXAMPLES_PER_TAG,

                "scope":
                    KEEP_CATEGORIES,

                "model":
                    BERT_MODEL,

                "seed":
                    SEED,

                "max_epochs":
                    MAX_EPOCHS,

                "early_stopping_patience":
                    STOP_AFTER_BAD_EPOCHS,

                "early_stopping_threshold":
                    EARLY_STOPPING_THRESHOLD,

                "early_stopping_metric":
                    "eval_loss",
            },
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    (
        SAVE_DIR
        / "metrics.json"
    ).write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    tag_scores.to_csv(
        SAVE_DIR / "per_tag_test.csv",
        index=False
    )

    category_scores.to_csv(
        SAVE_DIR
        / "per_category_test.csv"
    )

    # --------------------------------------------------------
    # Save split IDs
    # --------------------------------------------------------

    for name, rows in splits.items():

        clip_ids = sorted(
            clips.loc[
                rows,
                "clip_id"
            ].tolist()
        )

        (
            SAVE_DIR
            / f"split_{name}.json"
        ).write_text(
            json.dumps(
                clip_ids,
                indent=2
            )
        )

    # --------------------------------------------------------
    # Save training curves
    # --------------------------------------------------------

    save_curves(
        trainer.state.log_history,
        SAVE_DIR
        / "f1_curves.png"
    )

    print(
        f"\nsaved everything to "
        f"{SAVE_DIR}"
    )


if __name__ == "__main__":
    main()