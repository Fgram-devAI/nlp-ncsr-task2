from __future__ import annotations

from embeddings_utils import (
    format_neighbours,
    load_glove,
    load_w2v,
    print_side_by_side,
    top_k,
)


WORD = "student"
EXCLUDE_UNIVERSITY = ["university", "college", "undergraduate"]
EXCLUDE_K12 = ["kindergarten", "elementary", "pupil"]

# --- filter-based exclusion params ---
POOL_SIZE = 200
THRESHOLD = 0.50


def neighbours_minus(model, positive, negative, k: int = 10):
    """Return top-k of (sum(positive) - sum(negative)). None if any token OOV."""
    for w in list(positive) + list(negative):
        if w not in model.key_to_index:
            return None
    return model.most_similar(positive=list(positive), negative=list(negative), topn=k)


def filter_neighbours(model, word, exclusion_seeds, k=10,
                      pool_size=POOL_SIZE, threshold=THRESHOLD):
    """Top-k nearest to ``word`` after dropping anything semantically close to
    any of ``exclusion_seeds`` (cosine sim > threshold) or in the seed list.

    Used as an alternative to the vector-arithmetic exclusion: it stays in the
    populated region of the embedding space instead of subtracting the query
    vector into noise territory.
    """
    if word not in model.key_to_index:
        return None
    seeds_present = [s for s in exclusion_seeds if s in model.key_to_index]

    candidates = model.most_similar(word, topn=pool_size)
    seed_set = set(seeds_present)

    kept = []
    for cand_word, cand_sim in candidates:
        if cand_word in seed_set:
            continue
        max_sim_to_seed = max(model.similarity(cand_word, s) for s in seeds_present)
        if max_sim_to_seed > threshold:
            continue
        kept.append((cand_word, cand_sim))
        if len(kept) == k:
            break
    return kept


def run_filtered(label: str, w2v, glove, seeds: list[str]) -> None:
    print(f"\n=== {label} (FILTER method) ===")
    print(f"    drop neighbours whose max cos-sim to {seeds} > {THRESHOLD}")
    print_side_by_side(
        title_a="word2vec",
        rows_a=format_neighbours(filter_neighbours(w2v, WORD, seeds)),
        title_b="glove",
        rows_b=format_neighbours(filter_neighbours(glove, WORD, seeds)),
    )


def run_query(label: str, w2v, glove, positive, negative=()) -> None:
    print(f"\n=== {label} ===")
    if negative:
        print(f"    query: {positive} − {list(negative)}")
    else:
        print(f"    query: {positive}")

    w2v_n = neighbours_minus(w2v, positive, negative)
    glove_n = neighbours_minus(glove, positive, negative)

    if w2v_n is None:
        print("  [word2vec] OOV in query terms")
    if glove_n is None:
        print("  [glove]    OOV in query terms")

    print_side_by_side(
        title_a="word2vec",
        rows_a=format_neighbours(w2v_n),
        title_b="glove",
        rows_b=format_neighbours(glove_n),
    )


def main() -> None:
    print("Loading word2vec-google-news-300 ...")
    w2v = load_w2v()
    print("Loading glove-wiki-gigaword-300 ...")
    glove = load_glove()

    run_query(
        "baseline: top-10 nearest to 'student'",
        w2v, glove,
        positive=[WORD],
    )
    # --- minus university association: algebra + filter ---
    run_query(
        "minus university association [algebra: most_similar(positive=, negative=)]",
        w2v, glove,
        positive=[WORD],
        negative=EXCLUDE_UNIVERSITY,
    )
    run_filtered(
        "minus university association",
        w2v, glove, EXCLUDE_UNIVERSITY,
    )

    # --- minus K-12 / school-pupil association: algebra + filter ---
    run_query(
        "minus K-12 / school-pupil association [algebra: most_similar(positive=, negative=)]",
        w2v, glove,
        positive=[WORD],
        negative=EXCLUDE_K12,
    )
    run_filtered(
        "minus K-12 / school-pupil association",
        w2v, glove, EXCLUDE_K12,
    )


if __name__ == "__main__":
    main()
