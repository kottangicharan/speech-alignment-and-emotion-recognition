"""Shared code for the speech / text / fusion pipelines (TESS multimodal emotion recognition).

Blocks (see README / REPORT for the reasoning):
  Preprocessing      speech: 16 kHz mono, trim silence, peak-normalize, crop 3 s | text: lowercase, strip punct, RoBERTa BPE
  Feature extraction speech: frozen WavLM-Large hidden states (T x 1024)        | text: frozen RoBERTa-base (tokens x 768)
  Temporal/context   speech: learned layer weights + 2-layer BiLSTM + attn pool  | text: Transformer encoder layer + attn pool
  Fusion             cross-modal attention (speech frames -> text tokens) + gated fusion   (or late concat ablation)
  Classifier         MLP -> 7 emotions
Frozen features are extracted once and cached in cache/ so every experiment trains in minutes.
"""
import argparse
import csv
import json
import os
import random
import re
import statistics as st
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("TESS_DIR", ROOT / "data" / "TESS"))   # any layout; searched recursively
CACHE, RESULTS, CKPT = ROOT / "cache", ROOT / "Results", ROOT / "checkpoints"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "pleasant_surprise", "sad"]
LABEL_ALIASES = {"ps": "pleasant_surprise", "pleasant": "pleasant_surprise", "surprise": "pleasant_surprise",
                 "surprised": "pleasant_surprise"}
SPEECH_MODEL, TEXT_MODEL = "microsoft/wavlm-large", "roberta-base"
SR, MAX_SECONDS, TOP_DB = 16000, 3.0, 30
LAYER_GROUPS = 6           # 24 transformer layers averaged in 6 groups of 4 -> learned weights over groups (RAM: ~2.6 GB)
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def index_dataset():
    """All TESS clips, de-duplicated by file name (the Kaggle zip ships a nested duplicate copy)."""
    items = {}
    for p in sorted(DATA.rglob("*.wav")):
        parts = p.stem.split("_")
        if len(parts) < 3:
            continue
        label = LABEL_ALIASES.get(parts[-1].lower(), parts[-1].lower())
        items.setdefault(p.name.lower(), {
            # "OA_bite_neutral.wav" is a known TESS naming typo for OAF
            "id": p.stem.upper(), "path": str(p), "speaker": {"OA": "OAF"}.get(parts[0].upper(), parts[0].upper()),
            "word": "_".join(parts[1:-1]).lower(), "label": label,
        })
    items = sorted(items.values(), key=lambda x: x["id"])
    counts = {e: sum(i["label"] == e for i in items) for e in EMOTIONS}
    unknown = {i["label"] for i in items} - set(EMOTIONS)
    assert not unknown, f"unknown labels {unknown}"
    assert len(items) == 2800 and set(counts.values()) == {400}, f"expected 2800 clips / 400 per class, got {len(items)} {counts}"
    spk = {s: sum(i["speaker"] == s for i in items) for s in ("OAF", "YAF")}
    assert spk == {"OAF": 1400, "YAF": 1400}, f"speaker counts {spk}"
    return items


def sentence(item):
    # TESS has no transcript file: every clip says the carrier phrase + target word
    return f"say the word {item['word'].replace('_', ' ')}"


def clean_text(s):
    return re.sub(r"[^a-z' ]+", " ", s.lower()).strip()


def make_splits(items):
    """random_s{0,1,2}: stratified 70/15/15. spk_OAF2YAF / spk_YAF2OAF: train+val on one speaker, test on the other."""
    path = CACHE / "splits.json"
    if path.exists():
        return json.loads(path.read_text())
    from sklearn.model_selection import train_test_split
    ids, labels = [i["id"] for i in items], [i["label"] for i in items]
    splits = {}
    for s in SEEDS:
        tr, rest, _, y_rest = train_test_split(ids, labels, test_size=0.30, stratify=labels, random_state=s)
        va, te = train_test_split(rest, test_size=0.50, stratify=y_rest, random_state=s)
        splits[f"random_s{s}"] = {"train": tr, "val": va, "test": te}
    for a, b in [("OAF", "YAF"), ("YAF", "OAF")]:
        src = [i for i in items if i["speaker"] == a]
        tr, va = train_test_split([i["id"] for i in src], test_size=0.15,
                                  stratify=[i["label"] for i in src], random_state=0)
        splits[f"spk_{a}2{b}"] = {"train": tr, "val": va, "test": [i["id"] for i in items if i["speaker"] == b]}
    CACHE.mkdir(exist_ok=True)
    path.write_text(json.dumps(splits))
    return splits


# ----------------------------------------------------------------------------- frozen feature extraction (cached)
def load_audio(path):
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    y = y[: int(MAX_SECONDS * SR)]
    return y / (np.abs(y).max() + 1e-8)


@torch.no_grad()
def speech_features(items):
    """id -> float16 array (LAYER_GROUPS, T, 1024) of WavLM-Large hidden states."""
    path = CACHE / f"wavlm_large_g{LAYER_GROUPS}.pt"
    if path.exists():
        return torch.load(path)
    from transformers import AutoFeatureExtractor, WavLMModel
    fe = AutoFeatureExtractor.from_pretrained(SPEECH_MODEL)
    model = WavLMModel.from_pretrained(SPEECH_MODEL).to(DEVICE).eval()
    if DEVICE == "cuda":
        model.half()
    feats = {}
    for n, it in enumerate(items):
        x = fe(load_audio(it["path"]), sampling_rate=SR, return_tensors="pt").input_values.to(DEVICE)
        hs = model(x.half() if DEVICE == "cuda" else x, output_hidden_states=True).hidden_states[1:]  # 24 layers
        h = torch.stack(hs).squeeze(1)                                      # (24, T, 1024)
        h = h.view(LAYER_GROUPS, -1, *h.shape[1:]).mean(1)                  # (G, T, 1024)
        feats[it["id"]] = h.half().cpu()
        if n % 200 == 0:
            print(f"  WavLM features {n}/{len(items)}", flush=True)
    CACHE.mkdir(exist_ok=True)
    torch.save(feats, path)
    return feats


@torch.no_grad()
def text_features(items):
    """id -> float16 (tokens, 768) RoBERTa-base last hidden state (+ token strings for inspection)."""
    path = CACHE / "roberta_base.pt"
    if path.exists():
        return torch.load(path)
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(DEVICE).eval()
    by_sentence = {}
    for s in sorted({clean_text(sentence(i)) for i in items}):
        enc = tok(s, return_tensors="pt").to(DEVICE)
        by_sentence[s] = model(**enc).last_hidden_state.squeeze(0).half().cpu()
    feats = {i["id"]: by_sentence[clean_text(sentence(i))] for i in items}
    torch.save(feats, path)
    return feats


# ----------------------------------------------------------------------------- batching
def pad(seqs):
    """list of (..., T, D) -> padded (B, ..., Tmax, D) float32 + bool mask (B, Tmax) True = real frame."""
    lengths = [s.shape[-2] for s in seqs]
    out = torch.zeros(len(seqs), *seqs[0].shape[:-2], max(lengths), seqs[0].shape[-1])
    mask = torch.zeros(len(seqs), max(lengths), dtype=torch.bool)
    for b, (s, n) in enumerate(zip(seqs, lengths)):
        out[b, ..., :n, :] = s.float()
        mask[b, :n] = True
    return out, mask


def batches(ids, items_by_id, sfeat, tfeat, batch_size, shuffle):
    ids = list(ids)
    if shuffle:
        random.shuffle(ids)
    for k in range(0, len(ids), batch_size):
        chunk = ids[k:k + batch_size]
        b = {"ids": chunk, "y": torch.tensor([EMOTIONS.index(items_by_id[i]["label"]) for i in chunk])}
        if sfeat is not None:
            b["speech"], b["speech_mask"] = pad([sfeat[i] for i in chunk])
        if tfeat is not None:
            b["text"], b["text_mask"] = pad([tfeat[i] for i in chunk])
        yield {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in b.items()}


# ----------------------------------------------------------------------------- model blocks
class AttnPool(nn.Module):
    """Learned attention pooling over time/tokens (masked)."""
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, x, mask):
        w = self.score(x).squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
        return (w.unsqueeze(-1) * x).sum(1)


class SpeechEncoder(nn.Module):
    """Temporal modelling: softmax layer weights over WavLM groups -> proj -> 2-layer BiLSTM -> attention pool."""
    def __init__(self, in_dim=1024, hidden=256, dropout=0.3):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(LAYER_GROUPS))
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.pool = AttnPool(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, mask):                      # x (B, G, T, D)
        x = (self.layer_w.softmax(0)[None, :, None, None] * x).sum(1)
        x = self.proj(self.norm(x))
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        seq, _ = self.lstm(packed)
        seq, _ = nn.utils.rnn.pad_packed_sequence(seq, batch_first=True, total_length=x.shape[1])
        return seq, self.pool(seq, mask)


class TextEncoder(nn.Module):
    """Contextual modelling: proj -> 1 Transformer encoder layer (self-attention across tokens) -> attention pool."""
    def __init__(self, in_dim=768, hidden=256, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.ctx = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=dropout,
                                              batch_first=True)
        self.pool = AttnPool(hidden)
        self.out_dim = hidden

    def forward(self, x, mask):
        seq = self.ctx(self.proj(x), src_key_padding_mask=~mask)
        return seq, self.pool(seq, mask)


def classifier(in_dim, dropout=0.3):
    return nn.Sequential(nn.Linear(in_dim, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, len(EMOTIONS)))


class SpeechModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = SpeechEncoder()
        self.head = classifier(self.enc.out_dim)

    def forward(self, b):
        _, z = self.enc(b["speech"], b["speech_mask"])
        return self.head(z), {"temporal": z}


class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TextEncoder()
        self.head = classifier(self.enc.out_dim)

    def forward(self, b):
        _, z = self.enc(b["text"], b["text_mask"])
        return self.head(z), {"contextual": z}


class FusionModel(nn.Module):
    """gated: speech frames attend to text tokens (cross-modal attention), then a learned sigmoid gate mixes
    the speech and text vectors per dimension. concat: late fusion baseline [speech; text] -> linear."""
    def __init__(self, mode="gated", d=256, dropout=0.3):
        super().__init__()
        self.mode = mode
        self.s_enc, self.t_enc = SpeechEncoder(), TextEncoder()
        self.s_proj = nn.Linear(self.s_enc.out_dim, d)
        if mode == "gated":
            self.cross = nn.MultiheadAttention(d, num_heads=4, dropout=dropout, batch_first=True)
            self.cross_norm = nn.LayerNorm(d)
            self.pool = AttnPool(d)
            self.gate = nn.Linear(2 * d, d)
        else:
            self.mix = nn.Sequential(nn.Linear(self.s_enc.out_dim + self.t_enc.out_dim, d), nn.GELU())
        self.head = classifier(d)

    def forward(self, b):
        s_seq, s_vec = self.s_enc(b["speech"], b["speech_mask"])
        t_seq, t_vec = self.t_enc(b["text"], b["text_mask"])
        extra = {"temporal": s_vec, "contextual": t_vec}
        if self.mode == "gated":
            q = self.s_proj(s_seq)
            att, _ = self.cross(q, t_seq, t_seq, key_padding_mask=~b["text_mask"])
            s = self.pool(self.cross_norm(q + att), b["speech_mask"])     # text-aware speech vector
            g = torch.sigmoid(self.gate(torch.cat([s, t_vec], -1)))
            z = g * s + (1 - g) * t_vec
            extra["gate"] = g.mean(-1)                                     # 1 = all speech, 0 = all text
        else:
            z = self.mix(torch.cat([s_vec, t_vec], -1))
        extra["fusion"] = z
        return self.head(z), extra


def build_model(pipeline, fusion_mode="gated"):
    return {"speech": SpeechModel, "text": TextModel}.get(pipeline, lambda: FusionModel(fusion_mode))().to(DEVICE)


# ----------------------------------------------------------------------------- train / evaluate
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def speaker_normalize(sfeat, items):
    """Per-speaker standardization of WavLM features (mean/std over all of that speaker's frames, per layer
    group and dim). Uses no labels, only which clips share a speaker, so it is legal at test time too
    (unsupervised speaker adaptation). Removes the speaker's voice offset that hurts cross-speaker transfer."""
    by_spk = {}
    for i in items:
        by_spk.setdefault(i["speaker"], []).append(i["id"])
    for spk, ids in by_spk.items():
        s = sq = n = 0
        for i in ids:
            h = sfeat[i].float()                       # (G, T, D)
            s, sq, n = s + h.sum(1), sq + (h * h).sum(1), n + h.shape[1]
        mu = s / n
        sd = (sq / n - mu * mu).clamp_min(1e-6).sqrt()
        for i in ids:
            sfeat[i] = ((sfeat[i].float() - mu[:, None]) / sd[:, None]).half()
    return sfeat


NAME_SUFFIX = ""   # "_spknorm" when --spk_norm is on


def load_features(pipeline, items, spk_norm=False):
    sfeat = speech_features(items) if pipeline in ("speech", "fusion") else None
    if spk_norm and sfeat is not None:
        sfeat = speaker_normalize(sfeat, items)
    tfeat = text_features(items) if pipeline in ("text", "fusion") else None
    return sfeat, tfeat


@torch.no_grad()
def predict(model, ids, items_by_id, sfeat, tfeat, batch_size=64):
    model.eval()
    out = {"ids": [], "y": [], "prob": [], "emb": {}}
    for b in batches(ids, items_by_id, sfeat, tfeat, batch_size, shuffle=False):
        logits, extra = model(b)
        out["ids"] += b["ids"]
        out["y"].append(b["y"].cpu())
        out["prob"].append(logits.softmax(-1).cpu())
        for k, v in extra.items():
            out["emb"].setdefault(k, []).append(v.float().cpu())
    out["y"] = torch.cat(out["y"]).numpy()
    out["prob"] = torch.cat(out["prob"]).numpy()
    out["emb"] = {k: torch.cat(v).numpy() for k, v in out["emb"].items()}
    out["pred"] = out["prob"].argmax(1)
    return out


def set_suffix(spk_norm):
    global NAME_SUFFIX
    NAME_SUFFIX = "_spknorm" if spk_norm else ""


def run_name(pipeline, fusion_mode):
    return (pipeline if pipeline != "fusion" else f"fusion_{fusion_mode}") + NAME_SUFFIX


def train_one(pipeline, split_name, split, items_by_id, sfeat, tfeat, fusion_mode="gated",
              epochs=40, patience=8, lr=1e-3, batch_size=32, smoke=False):
    set_seed(int(split_name[-1]) if split_name.startswith("random") else 0)
    model = build_model(pipeline, fusion_mode)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad = -1.0, None, 0
    train_ids = split["train"][:64] if smoke else split["train"]
    for ep in range(1 if smoke else epochs):
        model.train()
        losses = []
        for b in batches(train_ids, items_by_id, sfeat, tfeat, batch_size, shuffle=True):
            logits, _ = model(b)
            loss = F.cross_entropy(logits, b["y"], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = predict(model, split["val"][:64] if smoke else split["val"], items_by_id, sfeat, tfeat)
        acc = float((val["pred"] == val["y"]).mean())
        print(f"  [{run_name(pipeline, fusion_mode)} {split_name}] epoch {ep + 1:2d} loss {st.mean(losses):.4f} val_acc {acc:.4f}", flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    CKPT.mkdir(exist_ok=True)
    torch.save({"state": best_state, "val_acc": best, "fusion_mode": fusion_mode},
               CKPT / f"{run_name(pipeline, fusion_mode)}_{split_name}.pt")
    return best


def train_main(pipeline):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="all", help="all | random | speaker | comma list e.g. random_s0,spk_OAF2YAF")
    ap.add_argument("--fusion", default="gated", choices=["gated", "concat"])
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 64 clips: checks the pipeline runs")
    ap.add_argument("--spk_norm", action="store_true", help="per-speaker feature standardization (speech/fusion)")
    a = ap.parse_args()
    set_suffix(a.spk_norm)
    items = index_dataset()
    items_by_id = {i["id"]: i for i in items}
    splits = select_splits(make_splits(items), a.splits)
    sfeat, tfeat = load_features(pipeline, items, a.spk_norm)
    for name, split in splits.items():
        acc = train_one(pipeline, name, split, items_by_id, sfeat, tfeat, a.fusion, smoke=a.smoke)
        print(f"{run_name(pipeline, a.fusion)} {name}: best val acc {acc:.4f}")
    print("done -> run test.py to evaluate on the test sets")


def select_splits(splits, which):
    if which == "all":
        return splits
    if which in ("random", "speaker"):
        prefix = "random" if which == "random" else "spk"
        return {k: v for k, v in splits.items() if k.startswith(prefix)}
    return {k: splits[k] for k in which.split(",")}


# ----------------------------------------------------------------------------- test + reporting
def test_main(pipeline):
    from sklearn.metrics import f1_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="all")
    ap.add_argument("--fusion", default="gated", choices=["gated", "concat"])
    ap.add_argument("--spk_norm", action="store_true", help="per-speaker feature standardization (speech/fusion)")
    a = ap.parse_args()
    set_suffix(a.spk_norm)
    items = index_dataset()
    items_by_id = {i["id"]: i for i in items}
    splits = select_splits(make_splits(items), a.splits)
    sfeat, tfeat = load_features(pipeline, items, a.spk_norm)
    name = run_name(pipeline, a.fusion)
    for split_name, split in splits.items():
        ck = CKPT / f"{name}_{split_name}.pt"
        if not ck.exists():
            print(f"skip {split_name}: no checkpoint {ck.name}")
            continue
        model = build_model(pipeline, a.fusion)
        model.load_state_dict(torch.load(ck)["state"])
        out = predict(model, split["test"], items_by_id, sfeat, tfeat)
        acc = float((out["pred"] == out["y"]).mean())
        f1 = float(f1_score(out["y"], out["pred"], average="macro"))
        per_class = f1_score(out["y"], out["pred"], average=None, labels=range(len(EMOTIONS)))
        d = RESULTS / name / split_name
        d.mkdir(parents=True, exist_ok=True)
        metrics = {"pipeline": name, "split": split_name, "test_acc": acc, "macro_f1": f1, "n_test": len(out["y"]),
                   "per_class_f1": dict(zip(EMOTIONS, map(float, per_class)))}
        if "gate" in out["emb"]:
            metrics["mean_gate_speech_weight"] = float(out["emb"]["gate"].mean())
        (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
        save_predictions(d / "predictions.csv", out, items_by_id)
        plot_confusion(out, d / "confusion.png", f"{name} | {split_name} | acc {acc:.3f}")
        np.savez_compressed(d / "embeddings.npz", ids=np.array(out["ids"]), y=out["y"],
                            spk=np.array([items_by_id[i]["speaker"] for i in out["ids"]]), **out["emb"])
        print(f"{name} {split_name}: test acc {acc:.4f}  macro-F1 {f1:.4f}")
    write_tables()
    plot_embeddings(name)
    write_failure_cases(name)


def save_predictions(path, out, items_by_id):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "speaker", "word", "true", "pred", "confidence", "correct"])
        for i, y, p, prob in zip(out["ids"], out["y"], out["pred"], out["prob"]):
            it = items_by_id[i]
            w.writerow([i, it["speaker"], it["word"], EMOTIONS[y], EMOTIONS[p], f"{prob[p]:.4f}", int(y == p)])


def plot_confusion(out, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(out["y"], out["pred"], labels=range(len(EMOTIONS)),
                                            display_labels=[e.replace("pleasant_surprise", "ps") for e in EMOTIONS],
                                            normalize="true", values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fmt(vals):
    return f"{st.mean(vals) * 100:.2f} ± {st.stdev(vals) * 100:.2f}" if len(vals) > 1 else f"{vals[0] * 100:.2f}"


def write_tables():
    """Results/accuracy_table.{md,csv} and per_class_f1.md from every metrics.json present."""
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*/*/metrics.json"))]
    if not rows:
        return
    order = ["speech", "text", "fusion_gated", "fusion_concat",
             "speech_spknorm", "fusion_gated_spknorm", "fusion_concat_spknorm"]
    pipes = sorted({r["pipeline"] for r in rows}, key=lambda p: order.index(p) if p in order else 99)
    cols = [("Random 70/15/15 (3 seeds)", lambda s: s.startswith("random")),
            ("Speaker OAF→YAF", lambda s: s == "spk_OAF2YAF"),
            ("Speaker YAF→OAF", lambda s: s == "spk_YAF2OAF"),
            ("Speaker-independent mean", lambda s: s.startswith("spk"))]
    md = ["# Test accuracy (%) / macro-F1 (%)", "",
          "| Model | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    csv_rows = []
    for p in pipes:
        cells = []
        for cname, f in cols:
            sel = [r for r in rows if r["pipeline"] == p and f(r["split"])]
            if not sel:
                cells.append("–")
                continue
            acc, mf1 = [r["test_acc"] for r in sel], [r["macro_f1"] for r in sel]
            cells.append(f"{_fmt(acc)} / {_fmt(mf1)}")
            csv_rows.append({"model": p, "split": cname, "acc_mean": st.mean(acc), "macro_f1_mean": st.mean(mf1), "runs": len(sel)})
        md.append(f"| {p} | " + " | ".join(cells) + " |")
    md += ["", "Chance = 14.29 % (7 balanced classes).", ""]
    md += ["# Per-class F1 (%), mean over runs", ""]
    for group, f in [("Random split", lambda s: s.startswith("random")), ("Speaker-independent", lambda s: s.startswith("spk"))]:
        md += [f"## {group}", "", "| Model | " + " | ".join(EMOTIONS) + " |", "|---" * (len(EMOTIONS) + 1) + "|"]
        for p in pipes:
            sel = [r for r in rows if r["pipeline"] == p and f(r["split"])]
            if sel:
                md.append(f"| {p} | " + " | ".join(f"{100 * st.mean(r['per_class_f1'][e] for r in sel):.1f}" for e in EMOTIONS) + " |")
        md.append("")
    gates = [r for r in rows if "mean_gate_speech_weight" in r]
    if gates:
        md += ["# Fusion gate (mean weight on speech, 1 = speech only)", "", "| Model | Split | gate |", "|---|---|---|"]
        md += [f"| {r['pipeline']} | {r['split']} | {r['mean_gate_speech_weight']:.3f} |" for r in gates]
    (RESULTS / "accuracy_table.md").write_text("\n".join(md), encoding="utf-8")
    with open(RESULTS / "accuracy_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)


def plot_embeddings(name):
    """t-SNE and UMAP of the learned representations (test set), color = emotion, marker = speaker,
    plus silhouette score as a number for cluster separability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    try:
        import warnings
        warnings.filterwarnings("ignore", module="umap")
        import umap
    except ImportError:
        umap = None
    plots = RESULTS / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    sil_rows = []
    for split_name in ["random_s0", "spk_OAF2YAF"]:
        f = RESULTS / name / split_name / "embeddings.npz"
        if not f.exists():
            continue
        z = np.load(f)
        spk = z["spk"]
        for block in [k for k in ("temporal", "contextual", "fusion") if k in z.files]:
            X, y = z[block], z["y"]
            sil = float(silhouette_score(X, y)) if len(set(y)) > 1 else float("nan")
            sil_rows.append((name, split_name, block, sil))
            methods = [("tsne", TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X))]
            if umap:
                methods.append(("umap", umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X)))
            for mname, P in methods:
                fig, ax = plt.subplots(figsize=(7, 6))
                for k, e in enumerate(EMOTIONS):
                    for s, mk in [("OAF", "o"), ("YAF", "^")]:
                        m = (y == k) & (spk == s)
                        ax.scatter(P[m, 0], P[m, 1], s=10, marker=mk, color=plt.cm.tab10(k),
                                   label=e if s == "OAF" else None, alpha=0.75)
                ax.set_title(f"{mname.upper()} | {name} | {block} block | {split_name} test | silhouette {sil:.3f}\n"
                             "o = OAF, ^ = YAF", fontsize=9)
                ax.legend(fontsize=8, markerscale=1.5)
                ax.set_xticks([]); ax.set_yticks([])
                fig.tight_layout()
                fig.savefig(plots / f"{mname}_{name}_{block}_{split_name}.png", dpi=130)
                plt.close(fig)
    if sil_rows:
        path = RESULTS / "silhouette.csv"
        old = [r for r in (list(csv.reader(open(path, encoding="utf-8")))[1:] if path.exists() else []) if r[0] != name]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "split", "block", "silhouette"])
            w.writerows(old + [list(r) for r in sil_rows])


def write_failure_cases(name, k=5):
    """Most confident mistakes per split family -> Results/<model>/failure_cases.csv (for error analysis)."""
    rows = []
    for f in sorted((RESULTS / name).glob("*/predictions.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if r["correct"] == "0":
                rows.append({"split": f.parent.name, **r})
    rows.sort(key=lambda r: -float(r["confidence"]))
    picked = [r for r in rows if r["split"].startswith("random")][:k] + [r for r in rows if r["split"].startswith("spk")][:k]
    if picked:
        with open(RESULTS / name / "failure_cases.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(picked[0]))
            w.writeheader()
            w.writerows(picked)
