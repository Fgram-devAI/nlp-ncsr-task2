from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import kagglehub
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ----------------------------------------------------------------------- #
#                         Tokenizer + Vocabulary                          #
# ----------------------------------------------------------------------- #


def tokenize(text: str) -> list[str]:
    """Lowercase + word/punct token split (parity with the reference script)."""
    return re.findall(r"\w+|[^\w\s]", str(text).lower())


@dataclass
class Vocab:
    itos: list[str]
    stoi: dict[str, int]

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def numericalize(self, tokens: list[str]) -> list[int]:
        unk = self.unk_idx
        return [self.stoi.get(t, unk) for t in tokens]


def build_vocab(texts: Iterable[str], min_freq: int = 10) -> Vocab:
    """Vocab from training texts. Tokens with freq < min_freq become <UNK>."""
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(tokenize(t))
    itos = [PAD_TOKEN, UNK_TOKEN] + [
        tok for tok, freq in counter.items() if freq >= min_freq
    ]
    stoi = {tok: i for i, tok in enumerate(itos)}
    return Vocab(itos=itos, stoi=stoi)


# ----------------------------------------------------------------------- #
#                          Dataset + DataLoader                           #
# ----------------------------------------------------------------------- #


class TextClassDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], self.labels[idx]


def make_collate(vocab: Vocab, max_words: int):
    """Build a collate_fn that tokenizes, numericalizes, pads/truncates to max_words."""
    pad_idx = vocab.pad_idx

    def collate(batch):
        texts, labels = zip(*batch)
        ids = [vocab.numericalize(tokenize(t)) for t in texts]
        ids = [
            seq[:max_words] if len(seq) >= max_words
            else seq + [pad_idx] * (max_words - len(seq))
            for seq in ids
        ]
        X = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        return X, y

    return collate


def make_loaders(
    X_train: list[str], y_train: list[int],
    X_test: list[str],  y_test: list[int],
    vocab: Vocab,
    max_words: int,
    batch_size: int,
    pin_memory: bool = False,
):
    """Return (train_loader, test_loader) wired up with the right collate_fn."""
    train_ds = TextClassDataset(X_train, y_train)
    test_ds = TextClassDataset(X_test,  y_test)
    collate = make_collate(vocab, max_words)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, pin_memory=pin_memory,
    )
    return train_loader, test_loader


# ----------------------------------------------------------------------- #
#                             AG News loader                              #
# ----------------------------------------------------------------------- #


AG_NEWS_SLUG = "amananandrai/ag-news-classification-dataset"
AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass(frozen=True)
class TextDataset:
    X_train: list[str]
    y_train: list[int]
    X_test:  list[str]
    y_test:  list[int]
    label_names: list[str]


@lru_cache(maxsize=1)
def load_ag_news() -> TextDataset:
    base = Path(kagglehub.dataset_download(AG_NEWS_SLUG))
    train_df = pd.read_csv(base / "train.csv")
    test_df  = pd.read_csv(base / "test.csv")

    def _join(df):
        texts = [f"{t} {d}" for t, d in zip(df["Title"], df["Description"])]
        labels = [int(c) - 1 for c in df["Class Index"]]
        return texts, labels

    X_tr, y_tr = _join(train_df)
    X_te, y_te = _join(test_df)
    return TextDataset(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        label_names=AG_NEWS_LABELS,
    )


# ----------------------------------------------------------------------- #
#                              IMDB loader                                #
# ----------------------------------------------------------------------- #


IMDB_SLUG = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
IMDB_LABELS = ["negative", "positive"]


@lru_cache(maxsize=1)
def load_imdb(test_frac: float = 0.20, seed: int = 42) -> TextDataset:
    """Load IMDB 50k. 80/20 split (deterministic via seed)."""
    base = Path(kagglehub.dataset_download(IMDB_SLUG))
    csvs = list(base.glob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV found in {base}")
    df = pd.read_csv(csvs[0])
    # column names: 'review', 'sentiment' ('positive' / 'negative')
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_test = int(len(df) * test_frac)
    test_df, train_df = df.iloc[:n_test], df.iloc[n_test:]

    def _to_lists(d):
        texts = d["review"].tolist()
        labels = [IMDB_LABELS.index(s) for s in d["sentiment"]]
        return texts, labels

    X_tr, y_tr = _to_lists(train_df)
    X_te, y_te = _to_lists(test_df)
    return TextDataset(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        label_names=IMDB_LABELS,
    )


# ----------------------------------------------------------------------- #
#                  Pretrained embedding matrix builder                    #
# ----------------------------------------------------------------------- #


def build_pretrained_embedding_matrix(
    vocab: Vocab,
    keyed_vectors,   # gensim KeyedVectors (any dim)
    embedding_dim: int,
) -> torch.Tensor:
    """Map each token in our vocab to its KV vector. Unknowns -> small random init.

    PAD remains zero-init (handled by nn.Embedding's padding_idx).
    Returns a tensor of shape (len(vocab), embedding_dim).
    """
    import numpy as np
    rng = np.random.default_rng(0)
    matrix = rng.normal(loc=0.0, scale=0.1, size=(len(vocab), embedding_dim)).astype("float32")
    matrix[vocab.pad_idx] = 0.0
    hits = 0
    for token, idx in vocab.stoi.items():
        if token in keyed_vectors.key_to_index:
            v = keyed_vectors[token]
            if v.shape[0] == embedding_dim:
                matrix[idx] = v
                hits += 1
    print(f"  embedding init coverage: {hits}/{len(vocab)} tokens "
          f"({100*hits/len(vocab):.1f}%) found in pre-trained vectors")
    return torch.from_numpy(matrix)
