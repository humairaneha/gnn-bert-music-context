"""
task4_contrastive.py -- Task 4: contrastive GNN-BERT alignment.

Learns a shared embedding space between audio graphs and their captions with
InfoNCE, then uses it for two things:

    RETRIEVAL      caption -> audio and audio -> caption, R@1/5/10
    ZERO-SHOT TAGS score every graph against the 53 tag names encoded as text,
                   with no supervised tag training at all, and compare against
                   the supervised Task 3 rows.

Reuses everything: the cached graphs from mtat_graphs.py, the frozen BERT text
cache from task3_fusion.py, and the metric code from task3_gnn_only.py, so the
zero-shot numbers are directly comparable to the supervised ones.

THE DUPLICATE-CAPTION PROBLEM, AND WHAT THIS FILE DOES ABOUT IT
--------------------------------------------------------------
MTAT has no captions. The text is a template over ~91 instrument tags, so many
clips share a byte-identical caption. That breaks vanilla InfoNCE twice:

  training    two clips with the same caption in one batch become each other's
              negatives, so the loss asks the model to separate two things it
              cannot distinguish, and gradients fight themselves.
              -> fixed by masking same-caption pairs out of the negatives.

  evaluation  "the correct clip" is not unique, so plain R@1 is capped at
              1/group_size no matter how good the model is.
              -> fixed by GROUP-AWARE relevance: a query scores a hit if ANY
                 clip sharing its caption appears in the top K. This is the
                 standard treatment when relevance is one-to-many, and it must
                 be stated in the report rather than quietly applied.

main() prints the caption-uniqueness statistics before training so the numbers
can be read with the right amount of scepticism. If uniqueness is very low, the
retrieval table is weak evidence however it is computed -- but the zero-shot tag
deliverable is unaffected, because it never asks "which clip is this".

    python src/task4_contrastive.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gnn import (
    BATCH_SIZE, DEVICE, EPOCHS, HIDDEN, LABEL_SPACE, LR, PATIENCE, SEED,
    WEIGHT_DECAY, GraphSAGE, find_best_thresholds, plot_tsne, score,
)
from task3_fusion import DATA_DIR, FEATURE, EDGE_POLICY, build_text_cache


# ================= CONFIG =================

RESULT_DIR = Path("results/task4_contrastive")

D_EMBED = 256                # shared space dimension
TEMPERATURE = 0.07           # InfoNCE tau, learnable from here
CONTRASTIVE_BATCH = 128      # bigger is better: N-1 in-batch negatives
MASK_DUPLICATE_CAPTIONS = True
GROUP_AWARE_RETRIEVAL = True
RECALL_K = (1, 5, 10)
N_QUALITATIVE = 10           # brief asks for 10

# Zero-shot: how a bare tag name is turned into a sentence for the text tower.
# The training captions all look like "A music clip featuring violin and
# strings.", so a zero-shot query phrased the same way sits in the same region
# of BERT's input space. A bare token like "classical" would not.
TAG_PROMPT = "A music clip that sounds {}."

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================= MODEL =================

class DualEncoder(nn.Module):
    """
    Two towers, no fusion. The graph tower is trainable; the text tower is a
    projection on top of frozen cached BERT.

    Both outputs are L2-normalized. That is not cosmetic: InfoNCE similarity is
    a dot product, so without normalization the loss can be reduced by growing
    vector magnitudes instead of aligning directions, and the temperature stops
    meaning anything.
    """

    def __init__(self, in_dim, hidden=HIDDEN, d=D_EMBED, temperature=TEMPERATURE):
        super().__init__()
        self.gnn = GraphSAGE(in_dim, hidden, d)          # classifier head unused
        self.g_proj = nn.Sequential(nn.Linear(hidden, d), nn.ReLU(), nn.Linear(d, d))
        self.t_proj = nn.Sequential(nn.Linear(768, d), nn.ReLU(), nn.Linear(d, d))
        # learned in log space so it stays positive under gradient descent
        self.log_temp = nn.Parameter(torch.tensor(np.log(temperature), dtype=torch.float))

    def encode_graph(self, batch):
        _, g, _ = self.gnn(batch, return_embedding=True)
        return F.normalize(self.g_proj(g), dim=1)

    def encode_text(self, cls):
        return F.normalize(self.t_proj(cls), dim=1)

    def forward(self, batch, cls):
        return self.encode_graph(batch), self.encode_text(cls)


def info_nce(g, t, log_temp, caption_ids=None):
    """
    Symmetric InfoNCE over in-batch negatives.

    The brief writes one direction; both are computed and averaged because
    retrieval is evaluated in both directions and a one-sided loss produces an
    asymmetric space.

    caption_ids lets same-caption pairs be masked out of the negatives. Without
    it, two clips with identical text are trained to repel each other -- the
    model is asked to separate two inputs that are literally the same string.
    """
    logits = g @ t.t() / log_temp.exp()               # [N, N]
    n = logits.size(0)
    target = torch.arange(n, device=logits.device)

    if caption_ids is not None:
        same = caption_ids[:, None] == caption_ids[None, :]
        same.fill_diagonal_(False)                     # keep the true positive
        logits = logits.masked_fill(same, float("-inf"))

    return 0.5 * (F.cross_entropy(logits, target) +
                  F.cross_entropy(logits.t(), target))


# ================= EMBEDDING =================

@torch.no_grad()
def embed_all(model, graphs, H, batch_size=256):
    """Encodes every graph and its caption once. Returns (G, T, y, track_ids, texts)."""
    model.eval()
    loader = DataLoader(graphs, batch_size=batch_size)
    G, T, Y, ids, texts = [], [], [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        cls = torch.stack([H[s][0] for s in batch.text]).to(DEVICE)   # [B, 768]
        g, t = model(batch, cls)
        G.append(g.cpu()); T.append(t.cpu())
        y = batch.y.float()
        Y.append((y.squeeze(1) if y.dim() == 3 else y).cpu())
        ids.extend(list(batch.track_id)); texts.extend(list(batch.text))
    return (torch.cat(G).numpy(), torch.cat(T).numpy(),
            torch.cat(Y).numpy(), ids, texts)


# ================= RETRIEVAL =================

def caption_stats(texts, label=""):
    c = Counter(texts)
    sizes = np.array(sorted(c.values(), reverse=True))
    shared = int(sizes[sizes > 1].sum())
    stats = {
        "clips": len(texts), "unique_captions": len(c),
        "unique_fraction": round(len(c) / len(texts), 4),
        "largest_group": int(sizes[0]),
        "clips_sharing_a_caption": shared,
        "shared_fraction": round(shared / len(texts), 4),
    }
    if label:
        print(f"[{label}] {len(texts)} clips, {len(c)} unique captions "
              f"({stats['unique_fraction']:.1%}) | largest group {sizes[0]} | "
              f"{stats['shared_fraction']:.1%} of clips share their caption")
    return stats


def retrieval_metrics(query, gallery, texts, ks=RECALL_K,
                      group_aware=GROUP_AWARE_RETRIEVAL):
    """
    R@K in one direction. query[i] is matched against every gallery row.

    group_aware=True counts a hit if ANY gallery item whose caption equals the
    query's caption lands in the top K. With one-to-one captions this reduces
    exactly to standard R@K; with duplicates it is the only measure that is not
    capped below 1 by the data.
    """
    sim = query @ gallery.T                            # both L2-normalized
    order = np.argsort(-sim, axis=1)

    text_arr = np.asarray(texts)
    out = {}
    for k in ks:
        topk = order[:, :k]
        if group_aware:
            hit = (text_arr[topk] == text_arr[:, None]).any(axis=1)
        else:
            hit = (topk == np.arange(len(query))[:, None]).any(axis=1)
        out[f"R@{k}"] = round(float(hit.mean()), 4)
    # Analytic floor at every k, not just k=1: the brief's comparison table puts
    # a random baseline against the contrastive model at R@5, so R@1 alone is
    # not enough to fill it in.
    for k in ks:
        out[f"random_R@{k}"] = round(min(1.0, k / len(gallery)), 6)
        out[f"lift_R@{k}"] = (round(out[f"R@{k}"] / out[f"random_R@{k}"], 1)
                              if out[f"random_R@{k}"] > 0 else None)
    return out


def random_baseline(n_query, n_gallery, texts, ks=RECALL_K, seed=SEED,
                    group_aware=GROUP_AWARE_RETRIEVAL):
    """
    B1 for retrieval: untrained random embeddings, scored through the identical
    ranking code.

    The analytic floor k/N assumes a uniformly random ranking, which is the
    right null. This measures it empirically instead, which also catches the
    case where duplicated captions inflate group-aware recall above the naive
    floor -- with one-to-one captions the two agree, and any gap is itself worth
    reporting.
    """
    rng = np.random.default_rng(seed)
    d = 64
    q = rng.normal(size=(n_query, d)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    g = rng.normal(size=(n_gallery, d)); g /= np.linalg.norm(g, axis=1, keepdims=True)
    out = retrieval_metrics(q, g, texts, ks, group_aware)
    return {f"R@{k}": out[f"R@{k}"] for k in ks}


def qualitative_examples(G, T, texts, ids, y, labels, n=N_QUALITATIVE, seed=SEED):
    """Query caption -> top-3 clips, with scores and the clips' true tags."""
    rng = np.random.default_rng(seed)
    # clip ids are STRINGS here (MusicCaps ytid / derived short id), never ints
    picks = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    sim = T[picks] @ G.T
    out = []
    for row, qi in zip(sim, picks):
        top = np.argsort(-row)[:3]
        out.append({
            "query_caption": texts[qi],
            "query_clip": str(ids[qi]),
            "retrieved": [{
                "clip": str(ids[j]),
                "similarity": round(float(row[j]), 4),
                "caption": texts[j],
                "caption_matches_query": texts[j] == texts[qi],
                "tags": [t for t, v in zip(labels, y[j]) if v],
            } for j in top],
        })
    return out


def human_eval_sheet(G, T, texts, ids, out_dir=RESULT_DIR, n=20, seed=SEED):
    """
    Writes the rating sheet for the human evaluation the brief requires:
    "minimum 5 listeners rate whether retrieved clip matches caption on
    scale [1, 5]".

    One row per (caption, top-1 retrieved clip) pair, with blank columns for
    five raters. The clip id is included so a listener can pull the audio; the
    similarity score is included for your analysis but should NOT be shown to
    raters, since knowing the model was confident biases the rating.

    This is the one Task 4 deliverable that code cannot produce on its own.
    """
    import csv
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    sim = T[picks] @ G.T

    path = Path(out_dir) / "human_eval_sheet.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "query_caption", "retrieved_clip_id",
                    "model_similarity", "rater1", "rater2", "rater3",
                    "rater4", "rater5", "mean_rating"])
        for i, (row, qi) in enumerate(zip(sim, picks), start=1):
            top = int(np.argmax(row))
            w.writerow([i, texts[qi], str(ids[top]), round(float(row[top]), 4),
                        "", "", "", "", "", ""])
    print(f"wrote {path} -- {n} pairs, 5 blank rater columns, scale 1-5")
    return path


# ================= ZERO-SHOT TAGGING =================

@torch.no_grad()
def zero_shot_scores(model, G, labels, H_tags):
    """
    Score every graph against every tag name embedded through the TEXT tower.

    No tag supervision is involved: the model has only ever seen instrument
    captions. Asking it about genre and mood words is genuine zero-shot
    transfer, and it may well land near chance -- which is a reportable result,
    not a failure. The comparison against the supervised Task 3 rows is the
    point of the deliverable.
    """
    model.eval()
    cls = torch.stack([H_tags[t][0] for t in labels]).to(DEVICE)
    t_emb = model.encode_text(cls).cpu().numpy()       # [K, d]
    sim = G @ t_emb.T                                  # [N, K], cosine in [-1, 1]
    return (sim + 1) / 2                               # -> [0, 1] for thresholding


# ================= MAIN =================

def graphs_for(split):
    p = DATA_DIR / f"{split}_{FEATURE}_{EDGE_POLICY}.pt"
    if not p.exists():
        p = DATA_DIR / f"{split}_{FEATURE}.pt"
    return torch.load(p, weights_only=False)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {DEVICE} | graphs: {FEATURE}_{EDGE_POLICY} | d={D_EMBED}\n")

    graphs = {s: graphs_for(s) for s in ("train", "val", "test")}
    space = json.loads(LABEL_SPACE.read_text())
    labels = space["genre"] + space["mood"]
    n_genre = space["n_genre"]

    # --- how trustworthy can the retrieval table be? ---
    stats = {s: caption_stats([g.text for g in graphs[s]], s) for s in graphs}
    json.dump(stats, open(RESULT_DIR / "caption_stats.json", "w"), indent=2)
    if stats["test"]["unique_fraction"] < 0.2:
        print("\n  WARNING: fewer than 20% of test captions are unique. Retrieval "
              "numbers below\n  are group-aware and still weak evidence -- say so "
              "in the report. The zero-shot\n  tag deliverable is unaffected.\n")

    H, mask, tokens = build_text_cache(list(graphs.values()))
    tag_texts = [TAG_PROMPT.format(t) for t in labels]

    # tag prompts are new strings, so they need their own cache pass
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE).eval()
    with torch.no_grad():
        enc = tok(tag_texts, padding="max_length", truncation=True,
                  max_length=32, return_tensors="pt").to(DEVICE)
        tag_H = bert(**enc).last_hidden_state.cpu()
    H_tags = {t: tag_H[i] for i, t in enumerate(labels)}
    del bert

    # --- train ---
    caption_id = {c: i for i, c in enumerate(sorted({g.text for g in graphs["train"]}))}
    model = DualEncoder(graphs["train"][0].x.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(graphs["train"], batch_size=CONTRASTIVE_BATCH, shuffle=True,
                        drop_last=True)
    val_loader = DataLoader(graphs["val"], batch_size=CONTRASTIVE_BATCH)

    best, wait = -1.0, 0
    history = {"train_loss": [], "val_R@5": [], "temperature": []}
    ckpt = RESULT_DIR / "best_dual_encoder.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(DEVICE)
            cls = torch.stack([H[s][0] for s in batch.text]).to(DEVICE)
            g, t = model(batch, cls)
            cid = (torch.tensor([caption_id.get(s, -i - 1)
                                 for i, s in enumerate(batch.text)], device=DEVICE)
                   if MASK_DUPLICATE_CAPTIONS else None)
            loss = info_nce(g, t, model.log_temp, cid)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        Gv, Tv, _, _, tv = embed_all(model, graphs["val"], H)
        r = retrieval_metrics(Tv, Gv, tv)              # caption -> audio
        history["train_loss"].append(train_loss)
        history["val_R@5"].append(r["R@5"])
        history["temperature"].append(float(model.log_temp.exp().item()))

        marker = ""
        if r["R@5"] > best:
            best, wait = r["R@5"], 0
            torch.save(model.state_dict(), ckpt)
            marker = "  <- saved"
        else:
            wait += 1
        print(f"epoch {epoch:3d} | loss {train_loss:.4f} | val R@5 {r['R@5']:.4f} | "
              f"tau {model.log_temp.exp().item():.4f}{marker}")
        if wait >= PATIENCE:
            print(f"early stopped at epoch {epoch}")
            break

    json.dump(history, open(RESULT_DIR / "training_history.json", "w"), indent=2)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    # --- retrieval on test, both directions ---
    G, T, y, ids, texts = embed_all(model, graphs["test"], H)
    results = {
        "caption_stats": stats,
        "group_aware": GROUP_AWARE_RETRIEVAL,
        "caption_to_audio": retrieval_metrics(T, G, texts),
        "audio_to_caption": retrieval_metrics(G, T, texts),
        "temperature": float(model.log_temp.exp().item()),
    }

    results["random_baseline"] = random_baseline(len(T), len(G), texts)

    print("\n" + "=" * 66)
    print(f"{'row':<24}{'R@1':>12}{'R@5':>12}{'R@10':>12}")
    print("-" * 66)
    rb = results["random_baseline"]
    print(f"{'B1: random embeddings':<24}{rb['R@1']:>12.4f}{rb['R@5']:>12.4f}"
          f"{rb['R@10']:>12.4f}")
    for name in ("caption_to_audio", "audio_to_caption"):
        r = results[name]
        print(f"{name:<24}{r['R@1']:>12.4f}{r['R@5']:>12.4f}{r['R@10']:>12.4f}")
    c = results["caption_to_audio"]
    print(f"{'lift over chance':<24}{str(c['lift_R@1']) + 'x':>12}"
          f"{str(c['lift_R@5']) + 'x':>12}{str(c['lift_R@10']) + 'x':>12}")
    print("=" * 66)

    json.dump(qualitative_examples(G, T, texts, ids, y, labels),
              open(RESULT_DIR / "qualitative_retrieval.json", "w"), indent=2)
    human_eval_sheet(G, T, texts, ids)

    # --- zero-shot tag prediction vs the supervised Task 3 model ---
    Gv, _, yv, _, _ = embed_all(model, graphs["val"], H)
    p_val = zero_shot_scores(model, Gv, labels, H_tags)
    thresholds = find_best_thresholds(p_val, yv)       # VAL only, as everywhere else
    p_test = zero_shot_scores(model, G, labels, H_tags)
    zs, _ = score(y, p_test, thresholds, n_genre)
    results["zero_shot_tagging"] = zs

    print(f"\nzero-shot tagging (no tag supervision):")
    print(f"  Macro-F1 {zs['macro_f1']:.4f} | AUC-PR {zs['auc_pr']:.4f} | "
          f"genre {zs['genre_macro_f1']:.4f} | mood {zs['mood_macro_f1']:.4f}")
    sup = Path("results/task3_gnn_only") / f"{FEATURE}_{EDGE_POLICY}" / "test_metrics.json"
    if sup.exists():
        s = json.loads(sup.read_text())["test_tuned"]
        print(f"  supervised Task 3: Macro-F1 {s['macro_f1']:.4f} | "
              f"AUC-PR {s['auc_pr']:.4f}  (gap {zs['auc_pr'] - s['auc_pr']:+.4f})")

    json.dump(results, open(RESULT_DIR / "test_metrics.json", "w"), indent=2)
    torch.save({"G": torch.from_numpy(G), "T": torch.from_numpy(T),
                "y": torch.from_numpy(y), "track_id": ids},
               RESULT_DIR / "shared_space.pt")
    plot_tsne(G, y, labels, n_genre,
              torch.cat([g.y for g in graphs["train"]]).sum(0).numpy(), RESULT_DIR)
    print(f"\nwrote {RESULT_DIR}/")
    return results


if __name__ == "__main__":
    main()