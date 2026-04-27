"""Exercise C.6 — repeat C.5 with frozen GloVe embeddings.

Same as C.5 (GloVe-6B-100d init) but the embedding layer is *frozen* —
``embedding.weight.requires_grad = False`` — so only the RNN cells and the
linear head are trained.

Expected effect on the parameter count: roughly **−2M trainable** (the entire
embedding matrix is excluded), so models become much "lighter" from an
optimizer's standpoint even though their nominal size is identical.

Run from project root:
    python -m part_c_rnn_classification.c6_glove_frozen
"""

from __future__ import annotations

from pathlib import Path

from .experiments import (
    GridParams, MODEL_CONFIGS, print_summary_table, run_grid, summarize,
)


SEEDS = [0, 1, 2]
OUT_PATH = Path(__file__).parent / "results" / "c6_glove_frozen.json"


def main() -> None:
    params = GridParams(
        max_words=25,
        epochs=15,
        batch_size=1024,
        embedding_dim=100,
        hidden_dim=64,
        learning_rate=1e-3,
        pretrained=True,
        frozen_embedding=True,        # ← only difference vs C.5
        dataset="ag_news",
    )
    results = run_grid(MODEL_CONFIGS, SEEDS, params, out_path=OUT_PATH)

    print()
    print("=" * 72)
    print("C.6 SUMMARY (GloVe init, FROZEN)")
    print("=" * 72)
    print_summary_table(
        summarize(results),
        ordering=[c["name"] for c in MODEL_CONFIGS],
    )


if __name__ == "__main__":
    main()
