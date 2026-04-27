"""Exercise C.7 — repeat C.1 on the IMDB 50k movie-review dataset.

Same 6 architectures × 3 seeds, same hyper-parameters as C.1, but the dataset
is **IMDB** (binary sentiment: negative / positive). Split: deterministic
80/20 of the 50k reviews (40k train / 10k test, seed=42 for the shuffle).

A few things to be aware of vs C.1:
- 2-class problem instead of 4 → ``num_classes=2`` (handled automatically
  in ``run_grid`` via ``len(ds.label_names)``).
- IMDB reviews are **long** (often 200-500 words). With ``MAX_WORDS=25`` we
  truncate aggressively, so accuracy will likely sit lower than AG News.
- IMDB vocab is much bigger than AG News at the same ``min_freq``.

First run downloads ~80 MB of CSV via kagglehub.

Run from project root:
    python -m part_c_rnn_classification.c7_imdb
"""

from __future__ import annotations

from pathlib import Path

from .experiments import (
    GridParams, MODEL_CONFIGS, print_summary_table, run_grid, summarize,
)


SEEDS = [0, 1, 2]
OUT_PATH = Path(__file__).parent / "results" / "c7_imdb.json"


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
        dataset="imdb",                # ← only difference vs C.1
    )
    results = run_grid(MODEL_CONFIGS, SEEDS, params, out_path=OUT_PATH)

    print()
    print("=" * 72)
    print("C.7 SUMMARY (IMDB 80/20)")
    print("=" * 72)
    print_summary_table(
        summarize(results),
        ordering=[c["name"] for c in MODEL_CONFIGS],
    )


if __name__ == "__main__":
    main()
