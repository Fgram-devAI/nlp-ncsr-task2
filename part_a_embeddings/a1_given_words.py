from __future__ import annotations

from embeddings_utils import (
    compare_lists,
    format_neighbours,
    load_glove,
    load_w2v,
    print_side_by_side,
    top_k,
)


WORDS = ["car", "jaguar", "Jaguar", "facebook"]


def run_for_word(word: str, w2v, glove) -> None:
    print(f"\n=== {word!r} ===")

    w2v_neighbours = top_k(w2v, word, k=10)
    glove_neighbours = top_k(glove, word, k=10)

    if w2v_neighbours is None:
        print(f"  [word2vec] OOV: {word!r}")
    if glove_neighbours is None:
        print(f"  [glove]    OOV: {word!r}")

    print_side_by_side(
        title_a="word2vec",
        rows_a=format_neighbours(w2v_neighbours),
        title_b="glove",
        rows_b=format_neighbours(glove_neighbours),
    )

    common, n_common = compare_lists(w2v_neighbours, glove_neighbours)
    print(f"\n  common words ({n_common}/10): {common}")


def main() -> None:
    print("Loading word2vec-google-news-300 ... (cached on disk after first run)")
    w2v = load_w2v()
    print("Loading glove-wiki-gigaword-300 ...")
    glove = load_glove()

    for word in WORDS:
        run_for_word(word, w2v, glove)


if __name__ == "__main__":
    main()
