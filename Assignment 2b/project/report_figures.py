"""Report figure: t-SNE grid of the temporal, contextual and fusion blocks (rows = random / unseen speaker).
Reads Results/<model>/<split>/embeddings.npz written by test.py -> Results/plots/tsne_grid.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

R = Path(__file__).parent / "Results"
EMO = ["angry", "disgust", "fear", "happy", "neutral", "pleasant surprise", "sad"]
COLS = [("speech", "temporal", "Temporal block (speech BiLSTM)"),
        ("text", "contextual", "Contextual block (text Transformer)"),
        ("fusion_gated", "fusion", "Fusion block (gated)")]
ROWS = [("random_s0", "Random split"), ("spk_OAF2YAF", "Unseen speaker (OAF→YAF)")]

fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
for r, (split, rname) in enumerate(ROWS):
    for c, (model, block, cname) in enumerate(COLS):
        z = np.load(R / model / split / "embeddings.npz")
        X, y, spk = z[block], z["y"], z["spk"]
        P = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
        ax = axes[r, c]
        for k in range(7):
            for s, mk in [("OAF", "o"), ("YAF", "^")]:
                m = (y == k) & (spk == s)
                ax.scatter(P[m, 0], P[m, 1], s=9, marker=mk, color=plt.cm.tab10(k), alpha=0.75)
        ax.set_title(f"{cname}\n{rname} · silhouette {silhouette_score(X, y):.2f}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
handles = [plt.Line2D([], [], ls="", marker="o", color=plt.cm.tab10(k), label=e) for k, e in enumerate(EMO)]
handles += [plt.Line2D([], [], ls="", marker=m, color="grey", label=f"{s} speaker") for s, m in [("OAF", "o"), ("YAF", "^")]]
fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=11, markerscale=1.6, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(R / "plots" / "tsne_grid.png", dpi=150)
print("wrote", R / "plots" / "tsne_grid.png")
