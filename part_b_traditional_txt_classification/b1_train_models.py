
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from data_utils import AGNews, load_ag_news


@dataclass
class TrainedModel:
    name: str
    vectorizer: TfidfVectorizer
    classifier: object
    accuracy: float
    vocab_size: int
    train_time_s: float
    y_pred_test: list[int]


def _word_vectorizer() -> TfidfVectorizer:
    # lowercase already done in data_utils, but enabling it here too is harmless.
    return TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 1))


def _char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, analyzer="char", ngram_range=(3, 3))


# (display name, vectorizer factory, classifier factory)
MODEL_CONFIGS: list[tuple[str, Callable[[], TfidfVectorizer], Callable[[], object]]] = [
    ("NB  (word 1-grams)", _word_vectorizer, lambda: MultinomialNB()),
    ("NB  (char 3-grams)", _char_vectorizer, lambda: MultinomialNB()),
    ("SVM (word 1-grams)", _word_vectorizer, lambda: LinearSVC(C=1.0)),
    ("SVM (char 3-grams)", _char_vectorizer, lambda: LinearSVC(C=1.0)),
]


def train_one(name, vectorizer_factory, classifier_factory, ds: AGNews) -> TrainedModel:
    vec = vectorizer_factory()
    clf = classifier_factory()

    t0 = time.perf_counter()
    X_train = vec.fit_transform(ds.X_train)
    clf.fit(X_train, ds.y_train)
    elapsed = time.perf_counter() - t0

    X_test = vec.transform(ds.X_test)
    y_pred = clf.predict(X_test).tolist()
    acc = accuracy_score(ds.y_test, y_pred)

    return TrainedModel(
        name=name,
        vectorizer=vec,
        classifier=clf,
        accuracy=acc,
        vocab_size=len(vec.vocabulary_),
        train_time_s=elapsed,
        y_pred_test=y_pred,
    )


def train_all(ds: AGNews | None = None) -> list[TrainedModel]:
    if ds is None:
        ds = load_ag_news()
    results = []
    for name, vec_f, clf_f in MODEL_CONFIGS:
        print(f"  training: {name} ...", flush=True)
        results.append(train_one(name, vec_f, clf_f, ds))
    return results


def print_table(results: list[TrainedModel]) -> None:
    header = f"{'metric':<16} | " + " | ".join(f"{r.name:>20}" for r in results)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    print(f"{'Accuracy':<16} | " +
          " | ".join(f"{r.accuracy:>20.4f}" for r in results))
    print(f"{'Dimensionality':<16} | " +
          " | ".join(f"{r.vocab_size:>20,}" for r in results))
    print(f"{'Time (sec)':<16} | " +
          " | ".join(f"{r.train_time_s:>20.2f}" for r in results))
    print(sep)


def main() -> None:
    print("Loading AG News ...")
    ds = load_ag_news()
    print(f"  train: {len(ds.X_train):>6}  test: {len(ds.X_test):>6}")
    print()
    results = train_all(ds)
    print()
    print_table(results)


if __name__ == "__main__":
    main()
