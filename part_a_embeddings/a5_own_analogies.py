from __future__ import annotations

from embeddings_utils import analogy, load_glove, load_w2v


# Each entry: (a, b, c) representing  a - b + c
# (with the casing word2vec expects; GloVe queries are lowercased automatically)
ANALOGIES = [
    ("dollar",     "USA",    "Greece"),
    ("basketball", "basket", "feet"),
    ("singer",       "actor",  "acting"),
]


def run_analogy(a: str, b: str, c: str, w2v, glove) -> None:
    print(f"\n=== {a} − {b} + {c} = ? ===")

    w2v_result = analogy(w2v, positive=[a, c], negative=[b], topn=2)
    glove_result = analogy(
        glove,
        positive=[a.lower(), c.lower()],
        negative=[b.lower()],
        topn=2,
    )

    def _print(label: str, result):
        if result is None:
            print(f"  {label:<10} OOV in query terms")
            return
        print(f"  {label}:")
        for word, sim in result:
            print(f"    {word:<20} {sim:6.3f}")

    _print("word2vec", w2v_result)
    _print("glove",    glove_result)


def main() -> None:
    print("Loading word2vec-google-news-300 ...")
    w2v = load_w2v()
    print("Loading glove-wiki-gigaword-300 ...")
    glove = load_glove()

    for a, b, c in ANALOGIES:
        run_analogy(a, b, c, w2v, glove)


if __name__ == "__main__":
    main()
