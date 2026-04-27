from __future__ import annotations

from collections import Counter

from b1_train_models import train_all
from data_utils import load_ag_news


def main() -> None:
    print("Loading AG News ...")
    ds = load_ag_news()

    print("Training the 4 models ...")
    results = train_all(ds)

    # Boolean per test doc: True iff every model mis-predicts it.
    n_test = len(ds.y_test)
    all_wrong = []
    for i in range(n_test):
        truth = ds.y_test[i]
        if all(r.y_pred_test[i] != truth for r in results):
            all_wrong.append(i)

    print(f"\nTest docs misclassified by ALL 4 models: {len(all_wrong)}"
          f" / {n_test} ({100*len(all_wrong)/n_test:.2f}%)")

    counts = Counter(ds.y_test[i] for i in all_wrong)
    print("\nPer-category counts of unanimously-wrong docs:")
    for c, name in enumerate(ds.label_names):
        print(f"  {name:<10} : {counts.get(c, 0)}")

    if all_wrong:
        idx = all_wrong[0]
        print("\n--- example unanimously misclassified document ---")
        print(f"  test index : {idx}")
        print(f"  true label : {ds.label_names[ds.y_test[idx]]}")
        for r in results:
            print(f"  {r.name:<22} predicted -> {ds.label_names[r.y_pred_test[idx]]}")
        print(f"\n  text:\n    {ds.X_test[idx]}")


if __name__ == "__main__":
    main()
