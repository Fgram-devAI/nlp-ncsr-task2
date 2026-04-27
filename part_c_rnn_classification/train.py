from __future__ import annotations

import time
from typing import Optional

import torch
from sklearn.metrics import accuracy_score
from torch import nn

from .model import count_parameters


def get_default_device() -> torch.device:
    """cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[list[int], list[int], float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    y_true, y_pred, losses = [], [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        losses.append(loss_fn(logits, y).item())
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
    return y_true, y_pred, sum(losses) / max(len(losses), 1)


def train_one_run(
    model: nn.Module,
    train_loader,
    test_loader,
    *,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
    verbose: bool = True,
    progress: bool = False,    # set True for tqdm bars in interactive notebooks
) -> dict:
    """Train ``model`` and evaluate on ``test_loader``. Returns result dict."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    train_losses: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        iterator = train_loader
        if progress:
            from tqdm import tqdm
            iterator = tqdm(train_loader, desc=f"  ep {epoch}/{epochs}", leave=False)
        for X, y in iterator:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg)
        if verbose:
            print(f"  epoch {epoch:>2}/{epochs}  train_loss={avg:.4f}", flush=True)
    elapsed = time.perf_counter() - t0
    sec_per_epoch = elapsed / epochs

    y_true, y_pred, test_loss = evaluate(model, test_loader, device)
    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "n_params": count_parameters(model),
        "train_time_s": float(elapsed),
        "sec_per_epoch": float(sec_per_epoch),
        "train_losses": [float(x) for x in train_losses],
        "test_loss": float(test_loss),
        "y_true": y_true,
        "y_pred": y_pred,
    }
