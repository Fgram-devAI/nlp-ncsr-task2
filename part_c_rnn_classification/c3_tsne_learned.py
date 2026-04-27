"""Exercise C.3 — t-SNE of the embeddings learned by a 1RNN model.

Trains a fresh 1RNN with the same hyper-parameters as C.1, then projects the
trained embedding-layer weights of the same 28 words used in A.6 (those that
appear in our model's vocabulary). Plots and saves the figure for direct visual
comparison with the GloVe t-SNE produced in A.6.

Run from project root:
    python -m part_c_rnn_classification.c3_tsne_learned
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from .data import build_vocab, load_ag_news, make_loaders
from .model import RecurrentClassifier
from .train import get_default_device, train_one_run


# Same word list as A.6 so the two diagrams are directly comparable.
WORDS_A6 = [
    "business", "career", "classroom", "company", "curriculum", "degree",
    "employee", "exam", "government", "homework", "investment", "job",
    "learning", "lecture", "lesson", "manager", "market", "office",
    "profession", "research", "salary", "school", "stock", "student",
    "teacher", "technology", "training", "university",
]


# Match C.1 baseline.
MAX_WORDS = 25
EPOCHS = 15
BATCH_SIZE = 1024
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
LEARNING_RATE = 1e-3
SEED = 0

OUT_DIR = Path(__file__).parent / "figures"
OUT_PATH = OUT_DIR / "tsne_rnn_learned_c3.png"


def main() -> None:
    device = get_default_device()
    print(f"device: {device}")

    # ---- data + vocab ----
    ds = load_ag_news()
    vocab = build_vocab(ds.X_train, min_freq=10)
    train_loader, test_loader = make_loaders(
        ds.X_train, ds.y_train, ds.X_test, ds.y_test,
        vocab, MAX_WORDS, BATCH_SIZE,
        pin_memory=(device.type == "cuda"),
    )

    # ---- train one 1RNN ----
    print("\nTraining a 1RNN (same config as C.1, single seed) ...")
    torch.manual_seed(SEED)
    model = RecurrentClassifier(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=4,
        rnn_type="rnn",
        num_layers=1,
        bidirectional=False,
        pad_idx=vocab.pad_idx,
    )
    result = train_one_run(
        model, train_loader, test_loader,
        epochs=EPOCHS, learning_rate=LEARNING_RATE,
        device=device, seed=SEED, verbose=True,
    )
    print(f"\ntrained 1RNN test accuracy: {result['accuracy']:.4f}")

    # ---- pull learned embeddings for the A.6 word list ----
    emb = model.embedding.weight.detach().cpu().numpy()   # (vocab, 100)
    present, missing = [], []
    for w in WORDS_A6:
        if w in vocab.stoi:
            present.append(w)
        else:
            missing.append(w)
    if missing:
        print(f"\nNote: {len(missing)} A.6 words not in our vocab (skipped): {missing}")
    print(f"plotting {len(present)} words present in vocab")

    vectors = np.stack([emb[vocab.stoi[w]] for w in present])
    print(f"vectors matrix: {vectors.shape}")

    # ---- t-SNE 2D ----
    print("Running t-SNE (perplexity=5) ...")
    coords = TSNE(
        n_components=2,
        perplexity=5,
        init="pca",
        random_state=42,
        learning_rate="auto",
    ).fit_transform(vectors)

    # ---- plot ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.7)
    for (x, y), word in zip(coords, present):
        ax.annotate(word, xy=(x, y), xytext=(5, 3),
                    textcoords="offset points", fontsize=10)
    ax.set_title(f"t-SNE of 1RNN-learned embeddings (100d) — {len(present)} of {len(WORDS_A6)} A.6 words")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"saved figure: {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
