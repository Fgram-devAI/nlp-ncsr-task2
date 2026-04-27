from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .data import (
    Vocab, build_vocab, load_ag_news, load_imdb, make_loaders,
    TextDataset, build_pretrained_embedding_matrix,
)
from .model import RecurrentClassifier
from .train import train_one_run, get_default_device


# The 6 architectures of question C.1.
MODEL_CONFIGS = [
    {"name": "1RNN",     "rnn_type": "rnn",  "num_layers": 1, "bidirectional": False},
    {"name": "1Bi-RNN",  "rnn_type": "rnn",  "num_layers": 1, "bidirectional": True},
    {"name": "2Bi-RNN",  "rnn_type": "rnn",  "num_layers": 2, "bidirectional": True},
    {"name": "1LSTM",    "rnn_type": "lstm", "num_layers": 1, "bidirectional": False},
    {"name": "1Bi-LSTM", "rnn_type": "lstm", "num_layers": 1, "bidirectional": True},
    {"name": "2Bi-LSTM", "rnn_type": "lstm", "num_layers": 2, "bidirectional": True},
]


@dataclass
class GridParams:
    max_words: int = 25
    epochs: int = 15
    batch_size: int = 1024
    embedding_dim: int = 100
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    min_freq: int = 10
    pretrained: bool = False                  # init from glove-6B-100d
    frozen_embedding: bool = False            # only relevant if pretrained=True
    dataset: str = "ag_news"                  # 'ag_news' | 'imdb'


def _load_dataset(name: str) -> TextDataset:
    if name == "ag_news":
        return load_ag_news()
    if name == "imdb":
        return load_imdb()
    raise ValueError(f"Unknown dataset {name!r}")


def _maybe_load_pretrained(vocab: Vocab, params: GridParams) -> Optional[torch.Tensor]:
    if not params.pretrained:
        return None
    assert params.embedding_dim == 100, (
        "glove-6B-100d gives 100d vectors; set embedding_dim=100 for pretrained=True"
    )
    print("  loading glove-wiki-gigaword-100 (cached after first call) ...")
    import gensim.downloader as api
    kv = api.load("glove-wiki-gigaword-100")
    return build_pretrained_embedding_matrix(vocab, kv, params.embedding_dim)


def run_grid(
    configs: list[dict],
    seeds: list[int],
    params: GridParams,
    *,
    device: Optional[torch.device] = None,
    out_path: Optional[Path] = None,
    verbose: bool = True,
) -> list[dict]:
    """Run each (config × seed) once, checkpointing to ``out_path`` after each run.

    On resume, runs already present in ``out_path`` are skipped.
    """
    device = device or get_default_device()
    print(f"device: {device}")
    print(f"params: {params}")

    # ---- data once ----
    ds = _load_dataset(params.dataset)
    print(f"dataset: {params.dataset}  train={len(ds.X_train)}  test={len(ds.X_test)}")
    vocab = build_vocab(ds.X_train, min_freq=params.min_freq)
    print(f"vocab size: {len(vocab)} (min_freq={params.min_freq})")
    train_loader, test_loader = make_loaders(
        ds.X_train, ds.y_train, ds.X_test, ds.y_test,
        vocab, params.max_words, params.batch_size,
        pin_memory=(device.type == "cuda"),
    )
    pretrained_emb = _maybe_load_pretrained(vocab, params)

    # ---- resume support ----
    results: list[dict] = []
    if out_path and out_path.exists():
        results = json.loads(out_path.read_text())
        print(f"resuming: {len(results)} prior runs in {out_path}")
    done = {(r["name"], r["seed"]) for r in results}

    for cfg in configs:
        for seed in seeds:
            key = (cfg["name"], seed)
            if key in done:
                print(f"  skip (already done): {cfg['name']} seed={seed}")
                continue

            print(f"\n▶ {cfg['name']}  seed={seed}  device={device}")
            torch.manual_seed(seed)
            model = RecurrentClassifier(
                vocab_size=len(vocab),
                embedding_dim=params.embedding_dim,
                hidden_dim=params.hidden_dim,
                num_classes=len(ds.label_names),
                rnn_type=cfg["rnn_type"],
                num_layers=cfg["num_layers"],
                bidirectional=cfg["bidirectional"],
                pretrained_emb=pretrained_emb,
                freeze_emb=params.frozen_embedding,
                pad_idx=vocab.pad_idx,
            )
            run_result = train_one_run(
                model, train_loader, test_loader,
                epochs=params.epochs,
                learning_rate=params.learning_rate,
                device=device,
                seed=seed,
                verbose=verbose,
            )
            record = {
                **cfg,
                "seed": seed,
                "device": device.type,
                **asdict(params),
                **run_result,
            }
            results.append(record)

            if out_path:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(results, indent=2))
                print(f"  ✓ saved checkpoint: {out_path} ({len(results)} runs)")

    return results


def summarize(results: list[dict]) -> dict:
    """Group by model name. Mean/std accuracy + sec/epoch + params."""
    import statistics

    by_name: dict[str, list[dict]] = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r)

    summary = {}
    for name, runs in by_name.items():
        accs = [r["accuracy"] for r in runs]
        spe = [r["sec_per_epoch"] for r in runs]
        summary[name] = {
            "mean_accuracy": round(statistics.mean(accs), 4),
            "std_accuracy":  round(statistics.stdev(accs), 4) if len(accs) > 1 else 0.0,
            "n_params":      runs[0]["n_params"],
            "sec_per_epoch": round(statistics.mean(spe), 2),
            "n_runs":        len(runs),
        }
    return summary


def print_summary_table(summary: dict, ordering: Optional[list[str]] = None) -> None:
    names = ordering or list(summary.keys())
    header = f"{'metric':<16} | " + " | ".join(f"{n:>10}" for n in names)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    print(f"{'Mean Accuracy':<16} | " +
          " | ".join(f"{summary[n]['mean_accuracy']:>10.4f}" for n in names))
    print(f"{'Std. Accuracy':<16} | " +
          " | ".join(f"{summary[n]['std_accuracy']:>10.4f}" for n in names))
    print(f"{'Parameters':<16} | " +
          " | ".join(f"{summary[n]['n_params']:>10,}" for n in names))
    print(f"{'Sec / epoch':<16} | " +
          " | ".join(f"{summary[n]['sec_per_epoch']:>10.2f}" for n in names))
    print(sep)
