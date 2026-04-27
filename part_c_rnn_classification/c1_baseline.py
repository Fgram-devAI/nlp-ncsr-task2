"""Exercise C.1 — baseline grid: 6 models × 3 seeds on AG News.

Default config matches the assignment exactly:
    EPOCHS=15  LEARNING_RATE=1e-3  BATCH_SIZE=1024
    EMBEDDING_DIM=100  HIDDEN_DIM=64  MAX_WORDS=25
    no pretrained embeddings, all params trainable

Results are checkpointed after every run to ``results/c1_baseline.json``,
so a Colab disconnect (or local Ctrl-C) can be resumed by re-running.

Run from project root:
    python -m part_c_rnn_classification.c1_baseline
"""

from __future__ import annotations

from pathlib import Path

from .experiments import (
    GridParams, MODEL_CONFIGS, print_summary_table, run_grid, summarize,
)


SEEDS = [0, 1, 2]
OUT_PATH = Path(__file__).parent / "results" / "c1_baseline.json"


def main() -> None:
    params = GridParams(
        max_words=25,
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
    print("C.1 SUMMARY")
    print("=" * 72)
    print_summary_table(
        summarize(results),
        ordering=[c["name"] for c in MODEL_CONFIGS],
    )


if __name__ == "__main__":
    main()
