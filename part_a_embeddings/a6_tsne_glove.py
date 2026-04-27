from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from embeddings_utils import load_glove


WORDS = [
    "business", "career", "classroom", "company", "curriculum", "degree",
    "employee", "exam", "government", "homework", "investment", "job",
    "learning", "lecture", "lesson", "manager", "market", "office",
    "profession", "research", "salary", "school", "stock", "student",
    "teacher", "technology", "training", "university",
]

OUT_DIR = Path(__file__).parent / "figures"
OUT_PATH = OUT_DIR / "tsne_glove_a6.png"


def main() -> None:
    print("Loading glove-wiki-gigaword-300 ...")
    glove = load_glove()

    missing = [w for w in WORDS if w not in glove.key_to_index]
    if missing:
        print(f"  WARNING: {len(missing)} OOV in GloVe: {missing}")
    present = [w for w in WORDS if w in glove.key_to_index]
    vectors = np.stack([glove[w] for w in present])
    print(f"  vectors matrix: {vectors.shape}")

    print("Running t-SNE (perplexity=5) ...")
    tsne = TSNE(
        n_components=2,
        perplexity=5,
        init="pca",
        random_state=42,
        learning_rate="auto",
    )
    coords_2d = tsne.fit_transform(vectors)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=40, alpha=0.7)
    for (x, y), word in zip(coords_2d, present):
        ax.annotate(
            word, xy=(x, y), xytext=(5, 3),
            textcoords="offset points", fontsize=10,
        )
    ax.set_title("t-SNE projection of GloVe (300d) vectors — 28 words")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"  saved figure: {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
