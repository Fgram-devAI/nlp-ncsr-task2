"""Exercise C.4 — repeat C.1 with MAX_WORDS=50 (vs 25 in the baseline).

Same 6 architectures × 3 seeds, identical hyper-parameters except for
``max_words=50``. Results saved to ``results/c4_max_words_50.json`` so the
notebook + report can show the C.1 vs C.4 comparison.

Run from project root:
    python -m part_c_rnn_classification.c4_max_words_50
"""

from __future__ import annotations

from pathlib import Path

from .experiments import (
    GridParams, MODEL_CONFIGS, print_summary_table, run_grid, summarize,
)


SEEDS = [0, 1, 2]
OUT_PATH = Path(__file__).parent / "results" / "c4_max_words_50.json"


def main() -> None:
    params = GridParams(
        max_words=50,                 # ← only difference vs C.1
        epochs=15,
        batch_size=1024,
        embedding_dim=100,
        hidden_dim=64,
        learning_rate=1e-3,
        pretrained=False,
        frozen_embedding=False,
        dataset="ag_news",
    )
    results = run_grid(MODEL_CONFIGS, SEEDS, params, out_path=OUT_PATH)

    print()
    print("=" * 72)
    print("C.4 SUMMARY (MAX_WORDS=50)")
    print("=" * 72)
    print_summary_table(
        summarize(results),
        ordering=[c["name"] for c in MODEL_CONFIGS],
    )


if __name__ == "__main__":
    main()
