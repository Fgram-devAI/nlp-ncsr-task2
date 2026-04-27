"""Exercise C.2 — CPU vs accelerator (GPU/MPS) timing comparison.

Trains 1RNN and 1LSTM on each of CPU and the available accelerator (MPS on
Apple Silicon, CUDA on NVIDIA). 5 epochs each — accuracy is irrelevant for the
question, we only care about sec/epoch.

Saves four runs to ``results/c2_cpu_vs_gpu.json`` and prints a comparison.

Run from project root:
    python -m part_c_rnn_classification.c2_cpu_vs_gpu
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .experiments import GridParams, run_grid


CONFIGS = [
    {"name": "1RNN",  "rnn_type": "rnn",  "num_layers": 1, "bidirectional": False},
    {"name": "1LSTM", "rnn_type": "lstm", "num_layers": 1, "bidirectional": False},
]
SEEDS = [0]
EPOCHS_FOR_TIMING = 5


def _accelerator() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    raise RuntimeError("No accelerator (CUDA or MPS) available; C.2 needs both CPU and accelerator")


def main() -> None:
    accel = _accelerator()
    print(f"\nCPU vs {accel.type.upper()} comparison\n")

    out_path = Path(__file__).parent / "results" / "c2_cpu_vs_gpu.json"
    if out_path.exists():
        out_path.unlink()  # always re-run cleanly

    params = GridParams(
        max_words=25,
        epochs=EPOCHS_FOR_TIMING,
        batch_size=1024,
        embedding_dim=100,
        hidden_dim=64,
        learning_rate=1e-3,
    )

    all_results = []
    for device in [torch.device("cpu"), accel]:
        print(f"\n--- device: {device.type.upper()} ---")
        # use a tagged out_path so checkpointing works per-device
        device_out = out_path.with_name(f"_c2_{device.type}.json")
        if device_out.exists():
            device_out.unlink()
        results = run_grid(
            CONFIGS, SEEDS, params,
            device=device,
            out_path=device_out,
            verbose=False,
        )
        all_results.extend(results)

    out_path.write_text(json.dumps(all_results, indent=2))

    # ---- summary table ----
    print()
    print("=" * 72)
    print(f"C.2 — sec/epoch  (CPU vs {accel.type.upper()})")
    print("=" * 72)
    header = f"{'model':<10} | {'CPU sec/ep':>12} | {accel.type.upper() + ' sec/ep':>12} | {'speed-up':>10}"
    sep = "-" * len(header)
    print(sep); print(header); print(sep)
    by = {(r["name"], r["device"]): r["sec_per_epoch"] for r in all_results}
    for name in ["1RNN", "1LSTM"]:
        cpu_t = by[(name, "cpu")]
        acc_t = by[(name, accel.type)]
        print(f"{name:<10} | {cpu_t:>12.2f} | {acc_t:>12.2f} | {cpu_t/acc_t:>9.1f}x")
    print(sep)


if __name__ == "__main__":
    main()
