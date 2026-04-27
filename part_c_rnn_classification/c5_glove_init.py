"""Exercise C.5 — repeat C.1 with embeddings initialized from GloVe-6B-100d.

Same 6 architectures × 3 seeds, identical hyper-parameters as C.1, except the
embedding layer's initial weights come from ``glove-wiki-gigaword-100``
(matched to our vocab; OOV tokens get small random init; <PAD> stays zero).
The embeddings are still **trainable** (frozen=False).

This requires ``EMBEDDING_DIM = 100`` (which already matches the C.1 baseline).
First run downloads ``glove-wiki-gigaword-100`` (~128 MB) via gensim into
``~/gensim-data/``; subsequent runs are cached.

Run from project root:
    python -m part_c_rnn_classification.c5_glove_init
"""

from __future__ import annotations

from pathlib import Path

from .experiments import (
    GridParams, MODEL_CONFIGS, print_summary_table, run_grid, summarize,
)


SEEDS = [0, 1, 2]
OUT_PATH = Path(__file__).parent / "results" / "c5_glove_init.json"


def main() -> None:
    params = GridParams(
        max_words=25,
        epochs=15,
        batch_size=1024,
        embedding_dim=100,            # required: glove-6B-100d is 100d
        hidden_dim=64,
        learning_rate=1e-3,
        pretrained=True,              # ← only difference vs C.1
        frozen_embedding=False,       # trainable
        dataset="ag_news",
    )
    results = run_grid(MODEL_CONFIGS, SEEDS, params, out_path=OUT_PATH)

    print()
    print("=" * 72)
    print("C.5 SUMMARY (GloVe init, trainable)")
    print("=" * 72)
    print_summary_table(
        summarize(results),
        ordering=[c["name"] for c in MODEL_CONFIGS],
    )


if __name__ == "__main__":
    main()
