"""Analyze MFA outputs in outputs/<variant>/ -> results/*.csv, results/summary.md, plots/*.png.

No gold labels exist, so quality is judged by: MFA's own alignment_analysis.csv (speech log-likelihood,
phone duration deviation), suspicious intervals (min-length phones, unknown words, very long words),
agreement of word boundaries across the 4 variants, and phonetic sanity on the ISLE minimal pairs.
"""
import csv
import itertools
import statistics as st
import wave
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from praatio import textgrid

ROOT = Path(__file__).parent
OUT, CORPUS, RES, PLOTS = ROOT / "outputs", ROOT / "corpus", ROOT / "results", ROOT / "plots"
MIN_PHONE = 0.0105            # MFA frame shift is 10 ms -> a 10 ms phone = squeezed/skipped
LONG_WORD = 1.0               # seconds; a word this long usually means absorbed silence or drift
MINIMAL_PAIRS = ["white", "bait", "new", "no", "bad", "bed"]
IPA_VOWELS = set("aeiouɑæɐɒɔəɛɜɪʊʌɝɚʉɨɵø")
SILENCE = {"", "sil", "sp", "<eps>"}


def is_vowel(p):
    return p[-1:].isdigit() or p[:1] in IPA_VOWELS   # ARPAbet vowels carry stress digits; MFA set is IPA


def load(tg_path):
    tg = textgrid.openTextgrid(str(tg_path), includeEmptyIntervals=False)
    tier = lambda kind: next(tg.getTier(n) for n in tg.tierNames if n.lower().endswith(kind))
    words = [(e.start, e.end, e.label.lower()) for e in tier("words").entries if e.label.lower() not in SILENCE]
    phones = [(e.start, e.end, e.label) for e in tier("phones").entries if e.label.lower() not in SILENCE]
    return words, phones


def boundary_diff_ms(w1, w2):
    """Mean |start|+|end| difference of the same word sequence in two alignments (ms)."""
    pairs = [(a, b) for a, b in zip(w1, w2) if a[2] == b[2]]
    if not pairs:
        return float("nan")
    return 1000 * st.mean(abs(a[0] - b[0]) / 2 + abs(a[1] - b[1]) / 2 for a, b in pairs)


def read_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(float), w.getframerate()


def plot(wav_path, words, phones, title, dst, window=None):
    x, sr = read_wav(wav_path)
    t0, t1 = window or (0, len(x) / sr)
    x = x[int(t0 * sr):int(t1 * sr)]
    fig, (ax_s, ax_w, ax_p) = plt.subplots(3, 1, figsize=(14, 6), sharex=True,
                                           gridspec_kw={"height_ratios": [4, 1, 1]})
    ax_s.specgram(x, Fs=sr, NFFT=512, noverlap=384, cmap="magma", xextent=(t0, t1))
    ax_s.set_ylim(0, 8000)
    ax_s.set_ylabel("Hz")
    ax_s.set_title(title)
    for ax, ivs, name in [(ax_w, words, "words"), (ax_p, phones, "phones")]:
        for s, e, lab in ivs:
            if e < t0 or s > t1:
                continue
            ax.axvline(s, color="k", lw=0.6)
            ax.axvline(e, color="k", lw=0.6)
            ax_s.axvline(s, color="w", lw=0.4, alpha=0.5 if name == "words" else 0.15)
            ax.text((s + e) / 2, 0.5, lab, ha="center", va="center", fontsize=8 if name == "words" else 6)
        ax.set_yticks([])
        ax.set_ylabel(name)
    ax_p.set_xlim(t0, t1)
    ax_p.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(dst, dpi=130)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows):
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def mfa_metrics(variant_dir):
    """Mean of MFA's alignment_analysis.csv columns (names vary slightly across versions)."""
    f = next(variant_dir.rglob("alignment_analysis.csv"), None)
    if not f:
        return {}
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    out = {}
    for col in ["speech_log_likelihood", "overall_log_likelihood", "phone_duration_deviation", "snr"]:
        vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "nan")]
        if vals:
            out[col] = st.mean(vals)
    return out


def main():
    RES.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    variants = sorted(d.name for d in OUT.iterdir() if d.is_dir())
    # data[variant][utt] = (words, phones)
    data = {v: {p.stem: load(p) for p in (OUT / v).rglob("*.TextGrid")} for v in variants}
    wavs = {p.stem: p for p in CORPUS.rglob("*.wav")}

    summary, anomalies, pairs_rows = [], [], []
    for v in variants:
        all_w = [w for ws, _ in data[v].values() for w in ws]
        all_p = [p for _, ps in data[v].values() for p in ps]
        short = [(u, p) for u, (_, ps) in data[v].items() for p in ps if p[1] - p[0] <= MIN_PHONE]
        unk = [(u, w) for u, (ws, _) in data[v].items() for w in ws if w[2] in ("<unk>", "spn")]
        long_w = [(u, w) for u, (ws, _) in data[v].items() for w in ws if w[1] - w[0] >= LONG_WORD]
        summary.append({
            "variant": v, "files": len(data[v]), "words": len(all_w), "phones": len(all_p),
            "mean_phone_ms": 1000 * st.mean(e - s for s, e, _ in all_p) if all_p else float("nan"),
            "min_len_phones": len(short), "unk_words": len(unk), "words_>=1s": len(long_w),
            **mfa_metrics(OUT / v),
        })
        for kind, items in [("min-length phone", short), ("unknown word", unk), ("very long word", long_w)]:
            for u, (s, e, lab) in items:
                anomalies.append({"variant": v, "file": u, "type": kind, "label": lab,
                                  "start": round(s, 3), "end": round(e, 3), "dur_ms": round(1000 * (e - s))})
        # ISLE minimal pairs: word + vowel durations
        for u, (ws, ps) in data[v].items():
            if not u.upper().startswith("ISLE"):
                continue
            for s, e, lab in ws:
                if lab in MINIMAL_PAIRS:
                    inside = [p for p in ps if p[0] >= s - 1e-6 and p[1] <= e + 1e-6]
                    vow = [p for p in inside if is_vowel(p[2])]
                    pairs_rows.append({"variant": v, "file": u, "word": lab,
                                       "word_ms": round(1000 * (e - s)), "phones": " ".join(p[2] for p in inside),
                                       "vowel": " ".join(p[2] for p in vow),
                                       "vowel_ms": round(1000 * sum(p[1] - p[0] for p in vow))})

    agreement = []
    for a, b in itertools.combinations(variants, 2):
        diffs = {u: boundary_diff_ms(data[a][u][0], data[b][u][0]) for u in data[a] if u in data[b]}
        agreement.append({"variant_a": a, "variant_b": b,
                          "mean_boundary_diff_ms": st.mean(diffs.values()) if diffs else float("nan"),
                          **{u: round(d, 1) for u, d in sorted(diffs.items())}})

    write_csv(RES / "summary.csv", summary)
    write_csv(RES / "boundary_agreement.csv", agreement)
    if pairs_rows:
        write_csv(RES / "minimal_pairs.csv", pairs_rows)
    if anomalies:
        write_csv(RES / "anomalies.csv", anomalies)
    md = ["# Alignment summary", "", "## Per variant", md_table(summary), "",
          "## Word-boundary agreement between variants (ms)", md_table(agreement), ""]
    if pairs_rows:
        md += ["## ISLE minimal pairs", md_table(pairs_rows), ""]
    md += [f"## Anomalies: {len(anomalies)} (see anomalies.csv)"]
    (RES / "summary.md").write_text("\n".join(md), encoding="utf-8")

    for v in variants:
        for u, (ws, ps) in data[v].items():
            window = (0, 6) if u.upper().startswith("F2B") else None  # 6 s zoom on long BU stories
            plot(wavs[u], ws, ps, f"{u}  [{v}]", PLOTS / f"{u}_{v}.png", window)
            if window:
                plot(wavs[u], ws, [], f"{u}  [{v}] full", PLOTS / f"{u}_{v}_full.png")
    print((RES / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    w = [(0.0, 0.5, "i"), (0.5, 1.0, "said")]
    assert boundary_diff_ms(w, w) == 0
    assert abs(boundary_diff_ms(w, [(0.02, 0.5, "i"), (0.5, 1.0, "said")]) - 5) < 1e-9
    assert is_vowel("AY1") and is_vowel("ɛ") and is_vowel("ʉː") and not is_vowel("B") and not is_vowel("tʃ")
    main()
