"""Train and compare MagnaTagATune text and audio–text models.

Modes:
    bert         BERT CLS classifier
    concat       GNN readout concatenated with BERT CLS
    crossattn    graph-query attention over BERT tokens
    mlp_bert     mean-pooled audio MLP with BERT CLS
    tags_linear  classifier on raw instrument-tag vectors

BERT is frozen and unique descriptions are cached. The audio and fusion
parameters are trained on genre and mood targets. Shared metrics and training
settings are imported from gnn.py. The MLP and tag-vector controls distinguish
representation choices from the information supplied by each modality.

    python src/train.py --task 3 --variant fusion
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import multilabel_confusion_matrix
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from gnn import (
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, HIDDEN, LABEL_SPACE, LR, NUM_LAYERS,
    PATIENCE, SEED, WEIGHT_DECAY, GraphSAGE,
    find_best_thresholds, make_tag_weights, plot_curves, plot_tsne,
    save_per_label_metrics, score,
)


# ================= CONFIG =================

DATA_DIR = Path("data/processed/mtat/graphs")
CACHE_DIR = Path("data/processed/mtat/bert_cache")
RESULT_DIR = Path("results/task3_fusion")
FEATURE = "mfcc"
EDGE_POLICY = "tau"              # which cached graph build to use

MODES = ["bert", "concat", "crossattn", "mlp_bert", "tags_linear"]

BERT_MODEL = "bert-base-uncased"
MAX_LEN = 32                     # the templated sentences are short
FREEZE_BERT = True               # see module docstring
D_MODEL = 128                    # both branches project here before fusion
N_CASE_STUDIES = 3               # brief asks for 3

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================= 1. FROZEN BERT TEXT CACHE =================

def build_text_cache(graph_sets, model_name=BERT_MODEL, max_len=MAX_LEN,
                     cache_dir=CACHE_DIR):
    """
    Encodes every UNIQUE text string once and returns
        (H[str] -> [max_len, 768], mask[str] -> [max_len], tokens[str] -> [str])

    The text is generated deterministically from each clip's instrument tags, so
    the number of distinct sentences is far smaller than the number of clips.
    Encoding per clip would repeat the same forward pass thousands of times.

    Token-level H is kept, not just CLS: cross-attention needs something to
    attend OVER. CLS alone would make the attention a no-op.
    """
    from transformers import AutoModel, AutoTokenizer

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_name.replace('/', '_')}_len{max_len}.pt"

    texts = sorted({g.text for gs in graph_sets for g in gs})
    if cache_path.exists():
        blob = torch.load(cache_path, weights_only=False)
        if set(blob["texts"]) == set(texts):
            print(f"BERT cache: reusing {cache_path} ({len(texts)} unique texts)")
            return blob["H"], blob["mask"], blob["tokens"]
        print("BERT cache: text set changed, re-encoding")

    n_clips = sum(len(gs) for gs in graph_sets)
    print(f"BERT cache: {len(texts)} unique texts across {n_clips} clips "
          f"({len(texts) / n_clips:.1%}) -- encoding once each")

    tok = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    H, mask, tokens = {}, {}, {}
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            enc = tok(chunk, padding="max_length", truncation=True,
                      max_length=max_len, return_tensors="pt").to(DEVICE)
            out = bert(**enc).last_hidden_state.cpu()          # [B, L, 768]
            am = enc["attention_mask"].cpu()
            for j, s in enumerate(chunk):
                H[s] = out[j].clone()
                mask[s] = am[j].clone()
                tokens[s] = tok.convert_ids_to_tokens(enc["input_ids"][j].cpu())

    torch.save({"texts": texts, "H": H, "mask": mask, "tokens": tokens}, cache_path)
    del bert
    print(f"BERT cache: wrote {cache_path}")
    return H, mask, tokens


def lookup_text(batch_texts, H, mask):
    """batch.text is a list of B strings; stack their cached tensors."""
    return (torch.stack([H[t] for t in batch_texts]),
            torch.stack([mask[t] for t in batch_texts]))


# ================= 2. INSTRUMENT MULTI-HOT (tags_linear control) =================

def text_to_tags(text: str, instrument):
    """
    Inverts build_text() from mtat_features.py:
        "A music clip featuring a, b and c."  ->  multi-hot over `instrument`

    Parsing rather than substring matching, because "guitar" is a substring of
    "electric guitar" and matching would set both.
    """
    idx = {t: i for i, t in enumerate(instrument)}
    v = np.zeros(len(instrument), dtype=np.float32)
    body = text.removeprefix("A music clip featuring ").removesuffix(".")
    if body in ("A music clip", ""):
        return v
    for part in body.replace(" and ", ", ").split(", "):
        if part in idx:
            v[idx[part]] = 1.0
    return v


# ================= 3. MODELS =================

class CrossAttention(nn.Module):
    """
    Single-query cross-attention. The graph readout g is ONE query attending
    over the L text tokens:

        A = softmax(g W_Q (H W_K)^T / sqrt(d))      [B, 1, L]
        context = A (H W_V)                          [B, d]

    Padded positions are masked to -inf before the softmax; without that, the
    attention spreads probability mass over [PAD] and the context vector is
    diluted by however much padding a short sentence happens to have.
    """

    def __init__(self, d_model=D_MODEL, text_dim=768):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(text_dim, d_model)
        self.v = nn.Linear(text_dim, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, g, H_text, text_mask, return_attention=False):
        Q = self.q(g).unsqueeze(1)                      # [B, 1, d]
        K, V = self.k(H_text), self.v(H_text)           # [B, L, d]
        scores = (Q @ K.transpose(1, 2)) / self.scale   # [B, 1, L]
        scores = scores.masked_fill(text_mask.unsqueeze(1) == 0, float("-inf"))
        A = torch.softmax(scores, dim=-1)
        context = (A @ V).squeeze(1)                    # [B, d]
        return (context, A.squeeze(1)) if return_attention else (context, None)


class FusionModel(nn.Module):
    """
    One class, five modes. Sharing the classifier head and the projection widths
    across modes is what keeps the ablation controlled -- only the branches
    present and the way they combine changes.
    """

    def __init__(self, mode, in_dim, n_labels, n_instrument=0,
                 hidden=HIDDEN, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.mode = mode

        if mode in ("concat", "crossattn"):
            self.gnn = GraphSAGE(in_dim, hidden, n_labels)   # its classifier is unused
            self.audio_proj = nn.Linear(hidden, d_model)
        elif mode == "mlp_bert":
            layers, d = [], in_dim
            for _ in range(NUM_LAYERS):
                layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = hidden
            self.mlp = nn.Sequential(*layers)
            self.audio_proj = nn.Linear(hidden, d_model)

        if mode in ("bert", "concat", "mlp_bert"):
            self.text_proj = nn.Linear(768, d_model)     # CLS -> d
        if mode == "crossattn":
            # CrossAttention projects the 768-d token states itself (W_K, W_V),
            # so no separate text_proj here
            self.cross = CrossAttention(d_model)
        if mode == "tags_linear":
            self.tag_proj = nn.Linear(n_instrument, d_model)

        z_dim = {"bert": d_model, "tags_linear": d_model,
                 "concat": 2 * d_model, "mlp_bert": 2 * d_model,
                 "crossattn": 2 * d_model}[mode]
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(z_dim, n_labels)

    def forward(self, batch, H_text, text_mask, tags=None,
                return_z=False, return_attention=False):
        attn = None

        if self.mode == "tags_linear":
            z = F.relu(self.tag_proj(tags))

        elif self.mode == "bert":
            z = F.relu(self.text_proj(H_text[:, 0]))          # CLS

        else:
            if self.mode == "mlp_bert":
                pooled = global_mean_pool(batch.x, batch.batch)
                g = self.mlp(pooled)
            else:
                # g is recomputed here, WITH gradients -- this is what makes the
                # fusion end-to-end rather than a frozen two-stage pipeline
                _, g, _ = self.gnn(batch, return_embedding=True)
            g = F.relu(self.audio_proj(g))

            if self.mode == "crossattn":
                context, attn = self.cross(g, H_text, text_mask, return_attention)
                z = torch.cat([g, context], dim=1)
            else:                                              # concat / mlp_bert
                t = F.relu(self.text_proj(H_text[:, 0]))
                z = torch.cat([g, t], dim=1)

        logits = self.classifier(self.dropout(z))
        if return_z or return_attention:
            return logits, z, attn
        return logits


# ================= 4. TRAIN / EVAL =================

def run_batch(model, batch, H, mask, instrument, **kw):
    """Assembles a batch's text tensors (and tags, if needed) and calls the model."""
    H_text, text_mask = lookup_text(batch.text, H, mask)
    H_text, text_mask = H_text.to(DEVICE), text_mask.to(DEVICE)
    tags = None
    if model.mode == "tags_linear":
        tags = torch.tensor(np.stack([text_to_tags(t, instrument) for t in batch.text]),
                            dtype=torch.float32, device=DEVICE)
    return model(batch, H_text, text_mask, tags, **kw)


def targets(batch):
    y = batch.y.float()
    return y.squeeze(1) if y.dim() == 3 else y


@torch.no_grad()
def predict(model, loader, H, mask, instrument, want_z=False):
    model.eval()
    ys, ps, zs = [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        out = run_batch(model, batch, H, mask, instrument, return_z=want_z)
        logits, z = (out[0], out[1]) if want_z else (out, None)
        ps.append(torch.sigmoid(logits).cpu())
        ys.append(targets(batch).cpu())
        if want_z:
            zs.append(z.cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return (y, p, torch.cat(zs).numpy()) if want_z else (y, p)


def train_one(mode, graphs, H, mask, instrument, labels, n_genre, groups,
              train_counts, out_root=RESULT_DIR):
    out_dir = Path(out_root) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\n{mode}\n{'=' * 72}")

    train_loader = DataLoader(graphs["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(graphs["val"], batch_size=BATCH_SIZE)
    test_loader = DataLoader(graphs["test"], batch_size=BATCH_SIZE)

    torch.manual_seed(SEED)          # identical init conditions for every mode
    model = FusionModel(mode, graphs["train"][0].x.shape[1], len(labels),
                        n_instrument=len(instrument)).to(DEVICE)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_par:,} (BERT frozen and cached separately)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=make_tag_weights(graphs["train"]).to(DEVICE))

    best, wait = -1.0, 0
    history = {k: [] for k in ("train_loss", "val_loss", "val_macro_f1",
                               "micro_f1", "auc_pr")}
    ckpt = out_dir / f"best_{mode}.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(run_batch(model, batch, H, mask, instrument),
                             targets(batch))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        y_val, p_val = predict(model, val_loader, H, mask, instrument)
        vm, _ = score(y_val, p_val, n_genre=n_genre)
        with torch.no_grad():
            val_loss = float(np.mean([
                criterion(run_batch(model, b.to(DEVICE), H, mask, instrument),
                          targets(b.to(DEVICE))).item() for b in val_loader]))

        for k, v in (("train_loss", train_loss), ("val_loss", val_loss),
                     ("val_macro_f1", vm["macro_f1"]), ("micro_f1", vm["micro_f1"]),
                     ("auc_pr", vm["auc_pr"])):
            history[k].append(v)

        marker = ""
        if vm["auc_pr"] > best:                    # same selection rule as every other row
            best, wait = vm["auc_pr"], 0
            torch.save(model.state_dict(), ckpt)
            marker = "  <- saved"
        else:
            wait += 1
        print(f"epoch {epoch:3d} | loss {train_loss:.4f}/{val_loss:.4f} | "
              f"val macroF1 {vm['macro_f1']:.4f} | AUC-PR {vm['auc_pr']:.4f}{marker}")
        if wait >= PATIENCE:
            print(f"early stopped at epoch {epoch}")
            break

    json.dump(history, open(out_dir / "training_history.json", "w"), indent=2)
    plot_curves(history, out_dir / "loss_f1_curves.png")

    # ---- evaluate: identical protocol to every other row ----
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    y_val, p_val = predict(model, val_loader, H, mask, instrument)
    thresholds = find_best_thresholds(p_val, y_val)              # VAL only
    np.save(out_dir / "thresholds.npy", thresholds)

    y_test, p_test, z_test = predict(model, test_loader, H, mask, instrument, want_z=True)
    at_half, _ = score(y_test, p_test, None, n_genre)
    tuned, pred = score(y_test, p_test, thresholds, n_genre)

    result = {"mode": mode, "trainable_params": n_par,
              "test_at_0.5": at_half, "test_tuned": tuned}
    json.dump(result, open(out_dir / "test_metrics.json", "w"), indent=2)

    save_per_label_metrics(y_test, pred, p_test, labels, thresholds, groups,
                           out_dir / "per_label_metrics.csv")
    np.save(out_dir / "label_confusion_matrices.npy",
            multilabel_confusion_matrix(y_test, pred))

    # t-SNE of the FUSED representation z, coloured by genre and by mood
    torch.save({"z": torch.from_numpy(z_test), "y": torch.from_numpy(y_test)},
               out_dir / "z_embeddings.pt")
    plot_tsne(z_test, y_test, labels, n_genre, train_counts, out_dir)

    print(f"test: macroF1 {tuned['macro_f1']:.4f} | AUC-PR {tuned['auc_pr']:.4f} | "
          f"genre {tuned['genre_macro_f1']:.4f} | mood {tuned['mood_macro_f1']:.4f}")
    return model, result, thresholds


# ================= 5. CASE STUDIES =================

@torch.no_grad()
def case_studies(model, graphs, H, mask, tokens, instrument, labels, thresholds,
                 n_case=N_CASE_STUDIES, out_dir=RESULT_DIR):
    """
    Export graph structure, token attention, and predictions for test examples.

    For each clip: the graph's edges (temporal chain plus similarity edges with
    their scores), the caption, which caption tokens the graph readout attended
    to, and predicted vs true tags. Only meaningful for crossattn -- the other
    modes have no attention to report.
    """
    model.eval()
    picks = graphs["test"][:n_case]
    loader = DataLoader(picks, batch_size=len(picks))
    batch = next(iter(loader)).to(DEVICE)
    logits, _, attn = run_batch(model, batch, H, mask, instrument,
                                return_attention=True)
    probs = torch.sigmoid(logits).cpu().numpy()

    studies = []
    for i, g in enumerate(picks):
        ei, ea = g.edge_index, g.edge_attr
        temporal, similarity = [], []
        seen = set()
        for k in range(ei.shape[1]):
            a, b = int(ei[0, k]), int(ei[1, k])
            if (b, a) in seen:
                continue
            seen.add((a, b))
            is_temp, sim = ea[k].tolist()
            (temporal if is_temp > .5 else similarity).append(
                {"from": a, "to": b, **({} if is_temp > .5 else {"similarity": round(sim, 3)})})

        toks = tokens[g.text]
        if attn is not None:
            w = attn[i].cpu().numpy()
            order = np.argsort(-w)
            top = [{"token": toks[j], "weight": round(float(w[j]), 4)}
                   for j in order[:8] if toks[j] not in ("[PAD]",)]
        else:
            top = None

        true = [t for t, v in zip(labels, g.y[0].tolist()) if v]
        predicted = [t for t, v in zip(labels, (probs[i] >= thresholds)) if v]
        studies.append({
            "track_id": g.track_id, "artist": g.artist, "text": g.text,
            "num_segments": int(g.num_segments),
            "temporal_edges": temporal, "similarity_edges": similarity,
            "top_attended_tokens": top,
            "true_tags": true, "predicted_tags": predicted,
            "correct": sorted(set(true) & set(predicted)),
            "missed": sorted(set(true) - set(predicted)),
            "false_positive": sorted(set(predicted) - set(true)),
        })

    path = Path(out_dir) / "case_studies.json"
    path.write_text(json.dumps(studies, indent=2))

    for s in studies:
        print(f"\n--- clip {s['track_id']} ({s['artist']}) ---")
        print(f"  text      {s['text']}")
        print(f"  graph     {s['num_segments']} segments, "
              f"{len(s['temporal_edges'])} temporal + "
              f"{len(s['similarity_edges'])} similarity edges")
        for e in s["similarity_edges"]:
            print(f"              segment {e['from']} <-> {e['to']}  sim={e['similarity']}")
        if s["top_attended_tokens"]:
            print("  attention " + ", ".join(
                f"{t['token']}({t['weight']:.3f})" for t in s["top_attended_tokens"][:6]))
        print(f"  true      {s['true_tags']}")
        print(f"  predicted {s['predicted_tags']}")
    print(f"\nwrote {path}")
    return studies


# ================= 6. MAIN =================

def graphs_for(split):
    path = DATA_DIR / f"{split}_{FEATURE}_{EDGE_POLICY}.pt"
    if not path.exists():
        path = DATA_DIR / f"{split}_{FEATURE}.pt"
    return torch.load(path, weights_only=False)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE} | graphs: {FEATURE}_{EDGE_POLICY} | "
          f"BERT: {BERT_MODEL} (frozen={FREEZE_BERT})\n")

    graphs = {s: graphs_for(s) for s in ("train", "val", "test")}
    print(f"graphs: " + ", ".join(f"{k}={len(v)}" for k, v in graphs.items()))

    space = json.loads(LABEL_SPACE.read_text())
    labels = space["genre"] + space["mood"]
    n_genre, instrument = space["n_genre"], space["instrument"]
    groups = ["genre"] * n_genre + ["mood"] * space["n_mood"]
    train_counts = torch.cat([g.y for g in graphs["train"]]).sum(0).numpy()
    print(f"targets: {n_genre} genre + {space['n_mood']} mood = {len(labels)}")
    print(f"instrument vocabulary (BERT input / tags_linear): {len(instrument)}\n")

    H, mask, tokens = build_text_cache(list(graphs.values()))

    results, crossattn_model, crossattn_thr = {}, None, None
    for mode in MODES:
        model, res, thr = train_one(mode, graphs, H, mask, instrument, labels,
                                    n_genre, groups, train_counts)
        results[mode] = res
        if mode == "crossattn":
            crossattn_model, crossattn_thr = model, thr

    json.dump(results, open(RESULT_DIR / "ablation_metrics.json", "w"), indent=2)

    print("\n" + "=" * 84)
    print(f"{'row':<16}{'params':>10}{'Macro-F1':>10}{'Micro-F1':>10}"
          f"{'AUC-PR':>10}{'genre F1':>11}{'mood F1':>10}")
    print("-" * 84)
    for mode in MODES:
        m = results[mode]["test_tuned"]
        print(f"{mode:<16}{results[mode]['trainable_params']:>10,}"
              f"{m['macro_f1']:>10.4f}{m['micro_f1']:>10.4f}{m['auc_pr']:>10.4f}"
              f"{m['genre_macro_f1']:>11.4f}{m['mood_macro_f1']:>10.4f}")
    print("=" * 84)

    # the two comparisons that decide what the table means
    if {"crossattn", "mlp_bert", "bert", "tags_linear"} <= results.keys():
        ap = {k: results[k]["test_tuned"]["auc_pr"] for k in results}
        print(f"\ngraph contribution   crossattn - mlp_bert = "
              f"{ap['crossattn'] - ap['mlp_bert']:+.4f}")
        print(f"                     (positive => the GRAPH adds something over "
              f"mean-pooled audio)")
        print(f"BERT contribution    bert - tags_linear     = "
              f"{ap['bert'] - ap['tags_linear']:+.4f}")
        print(f"                     (near zero => BERT adds nothing over the raw "
              f"instrument tags,\n                      which is the expected result "
              f"for template-generated text)")
        print(f"audio contribution   crossattn - bert       = "
              f"{ap['crossattn'] - ap['bert']:+.4f}")
        print(f"fusion strategy      crossattn - concat     = "
              f"{ap['crossattn'] - ap['concat']:+.4f}")

    if crossattn_model is not None:
        case_studies(crossattn_model, graphs, H, mask, tokens, instrument,
                     labels, crossattn_thr)

    print(f"\nwrote {RESULT_DIR}/")
    return results


if __name__ == "__main__":
    main()
