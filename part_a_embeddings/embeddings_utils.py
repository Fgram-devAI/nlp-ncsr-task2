from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import gensim.downloader as gensim_api
from gensim.models import KeyedVectors


W2V_NAME = "word2vec-google-news-300"
GLOVE_NAME = "glove-wiki-gigaword-300"


@lru_cache(maxsize=1)
def load_w2v() -> KeyedVectors:
    return gensim_api.load(W2V_NAME)


@lru_cache(maxsize=1)
def load_glove() -> KeyedVectors:
    return gensim_api.load(GLOVE_NAME)


def top_k(model: KeyedVectors, word: str, k: int = 10) -> list[tuple[str, float]] | None:
    """Top-K most-similar neighbours of ``word``. Returns ``None`` if OOV."""
    if word not in model.key_to_index:
        return None
    return model.most_similar(word, topn=k)


def compare_lists(
    list_a: Sequence[tuple[str, float]] | None,
    list_b: Sequence[tuple[str, float]] | None,
) -> tuple[list[str], int]:
    """Intersection of the words in two ranked lists. Empty if either is None."""
    if list_a is None or list_b is None:
        return [], 0
    set_b = {w for w, _ in list_b}
    common = [w for w, _ in list_a if w in set_b]
    return common, len(common)


def analogy(
    model: KeyedVectors,
    positive: Iterable[str],
    negative: Iterable[str],
    topn: int = 2,
) -> list[tuple[str, float]] | None:
    """Solve ``sum(positive) - sum(negative)``. Returns ``None`` if any token OOV.

    gensim's ``most_similar`` already excludes the input words from the result,
    so the top-N here is the top-N *new* candidates.
    """
    pos = list(positive)
    neg = list(negative)
    for w in pos + neg:
        if w not in model.key_to_index:
            return None
    return model.most_similar(positive=pos, negative=neg, topn=topn)


def format_neighbours(
    neighbours: Sequence[tuple[str, float]] | None,
    width: int = 22,
) -> list[str]:
    """Return one fixed-width line per neighbour (word + score), padded."""
    if neighbours is None:
        return ["<OOV>".ljust(width)]
    return [f"{w[:width-8]:<{width-8}} {s:6.3f}" for w, s in neighbours]


def print_side_by_side(
    title_a: str,
    rows_a: list[str],
    title_b: str,
    rows_b: list[str],
    col_width: int = 22,
    gap: str = "    ",
) -> None:
    """Print two columns of strings side-by-side with aligned headers."""
    header = f"{title_a:<{col_width}}{gap}{title_b:<{col_width}}"
    print(header)
    print("-" * len(header))
    for i in range(max(len(rows_a), len(rows_b))):
        left = rows_a[i] if i < len(rows_a) else " " * col_width
        right = rows_b[i] if i < len(rows_b) else ""
        print(f"{left:<{col_width}}{gap}{right}")
