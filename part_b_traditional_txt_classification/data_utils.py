from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import kagglehub
import pandas as pd


KAGGLE_SLUG = "amananandrai/ag-news-classification-dataset"

# Class Index → human label (Kaggle uses 1-indexed).
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass(frozen=True)
class AGNews:
    X_train: list[str]
    y_train: list[int]   # 0-indexed
    X_test:  list[str]
    y_test:  list[int]
    label_names: list[str]


def _join_text(title: str, desc: str) -> str:
    return f"{title} {desc}".lower()


def _load_split(csv_path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(csv_path)
    texts = [_join_text(t, d) for t, d in zip(df["Title"], df["Description"])]
    # convert 1..4 -> 0..3
    labels = [int(c) - 1 for c in df["Class Index"]]
    return texts, labels


@lru_cache(maxsize=1)
def load_ag_news() -> AGNews:
    """Load AG News (downloading via kagglehub on first call)."""
    base = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    X_train, y_train = _load_split(base / "train.csv")
    X_test,  y_test  = _load_split(base / "test.csv")
    return AGNews(
        X_train=X_train, y_train=y_train,
        X_test=X_test,  y_test=y_test,
        label_names=LABEL_NAMES,
    )


if __name__ == "__main__":
    ds = load_ag_news()
    print(f"train: {len(ds.X_train):>6} docs")
    print(f"test : {len(ds.X_test):>6} docs")
    print(f"labels: {ds.label_names}")
    print(f"\nsample train doc (label={ds.label_names[ds.y_train[0]]}):")
    print(f"  {ds.X_train[0][:200]}")
