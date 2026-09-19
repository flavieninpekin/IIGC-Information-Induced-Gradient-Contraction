"""Supervised cross-setting audit on DICES-350.

DICES-350: 350 adversarial conversations, each rated by all 123 raters, with
rater demographics. This script treats a rater demographic (race, age, or
gender) as the hidden condition and audits the condition-conditioned gradients
of a shared classifier at its fitted parameters, under the canonical kappa
definitions.

What it measures
----------------
1. Gradient audit at the shared model:
   - per-group mean gradient of the binary cross-entropy on the training rows;
   - per-item gradients for the within-group (episode) noise;
   - kappa_mix, kappa_ep, E_shared, E_contrast, sigma2 via
     ``iigc.metrics.kappa``.

2. Capacity comparison on held-out items:
   - shared model (one classifier, no group information);
   - group-conditioned models (one classifier per group, oracle group at
     training time);
   - average and worst-group accuracy per axis.

Protocol notes
--------------
- The split is by ``item_id``, never by row, so a conversation cannot appear in
  both train and test.
- ``Q_overall`` is the rater's perceived-harm judgment. ``Unsure`` labels are
  dropped and the remaining binary task is ``Yes`` (harmful) vs ``No``.
- Feature extraction happens once on the training text; no hyperparameter
  search is performed.

Usage
-----
  python run_dices_group_audit.py [--csv PATH] [--seeds 41 42 43] [--axes race age gender]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from iigc.metrics.kappa import (
    condition_decomposition,
    episode_decomposition,
    measurement_metadata,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_CSV = os.path.join(ROOT, "data", "external", "dices",
                          "diverse_safety_adversarial_dialog_350.csv")
OUT = os.path.join(ROOT, "data", "kappa", "dices350", "audit.json")

LABEL_MAP = {"No": 0, "Yes": 1}
AXES = {
    "race": ["White", "Black/African American", "Asian/Asian subcontinent"],
    "age": ["gen z", "millenial", "gen x+"],
    "gender": ["Woman", "Man"],
}
TEST_FRACTION = 0.30


def load(csv_path):
    cols = ["item_id", "rater_id", "rater_race", "rater_age", "rater_gender",
            "context", "response", "Q_overall"]
    df = pd.read_csv(csv_path, usecols=cols)
    raw_rows = len(df)
    df = df[df.Q_overall.isin(LABEL_MAP)].copy()
    df["label"] = df.Q_overall.map(LABEL_MAP)
    df["text"] = (df.context.astype(str).str.strip() + " \n "
                  + df.response.astype(str).str.strip())
    return df, raw_rows


def split_items(df, seed, fraction):
    items = np.sort(df.item_id.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    n_test = int(round(len(items) * fraction))
    test_items = set(items[:n_test])
    train_items = set(items[n_test:])
    train = df.item_id.isin(train_items).to_numpy()
    test = df.item_id.isin(test_items).to_numpy()
    return train, test, len(train_items), len(test_items)


def fit_logreg(X, y):
    clf = LogisticRegression(C=1.0, max_iter=3000, solver="liblinear")
    clf.fit(X, y)
    return clf


def per_group_mean_gradient(X, y, clf):
    p = clf.predict_proba(X)[:, 1]
    residual = p - y
    grad_w = np.asarray(X.T @ residual).ravel() / max(len(y), 1)
    grad_b = float(residual.mean())
    return np.concatenate([grad_w, [grad_b]])


def per_item_gradients(X, y, clf, item_ids):
    """Per-item mean gradients for one condition (episode = item)."""
    X = X.tocsr()
    p = clf.predict_proba(X)[:, 1]
    residual = p - y
    weighted = X.multiply(residual[:, np.newaxis]).tocsr()
    unique_items, inverse = np.unique(item_ids, return_inverse=True)
    n_items = len(unique_items)
    n_rows = X.shape[0]
    indicator = sp.csr_matrix(
        (np.ones(n_rows), (inverse, np.arange(n_rows))),
        shape=(n_items, n_rows))
    sums = indicator @ weighted
    counts = np.asarray(indicator.sum(axis=1)).ravel()
    bias_sums = np.asarray(indicator @ residual).ravel()
    keep = counts > 0
    dense = np.asarray(sums.todense())[keep] / counts[keep, None]
    bias = (bias_sums[keep] / counts[keep])[:, None]
    return np.concatenate([dense, bias], axis=1), counts[keep]


def accuracy_by_group(X, y, group, clf, groups):
    pred = clf.predict(X)
    rows = {}
    accs = []
    for g in groups:
        mask = group == g
        if not mask.any():
            continue
        acc = float((pred[mask] == y[mask]).mean())
        rows[g] = {"n": int(mask.sum()), "accuracy": acc}
        accs.append(acc)
    return rows, float(np.mean(accs)) if accs else float("nan"), \
        float(np.min(accs)) if accs else float("nan")


def audit_axis(df, X, train_mask, test_mask, axis, groups, shared_clf):
    """Gradient audit plus capacity comparison for one condition axis."""
    gcol = f"rater_{axis}"
    train_df = df[train_mask]
    test_df = df[test_mask]
    X_train = X[train_mask]
    X_test = X[test_mask]

    # ---- gradient audit at the shared model ----
    mean_grads, item_grads = [], []
    for g in groups:
        mask = (train_df[gcol] == g).to_numpy()
        Xg, yg = X_train[mask], train_df.label.to_numpy()[mask]
        mean_grads.append(per_group_mean_gradient(Xg, yg, shared_clf))
        item_g, _ = per_item_gradients(Xg, yg, shared_clf,
                                       train_df.item_id.to_numpy()[mask])
        item_grads.append(item_g)
    weights = [1.0 / len(groups)] * len(groups)
    structural = condition_decomposition(mean_grads, weights)
    noisy = episode_decomposition(item_grads, weights)

    # ---- capacity comparison on held-out items ----
    y_test = test_df.label.to_numpy()
    shared_rows, shared_avg, shared_worst = accuracy_by_group(
        X_test, y_test, test_df[gcol].to_numpy(), shared_clf, groups)
    conditional_rows, conditional_avg, conditional_worst = {}, [], []
    group_clfs = {}
    for g in groups:
        mask = (train_df[gcol] == g).to_numpy()
        clf = fit_logreg(X_train[mask], train_df.label.to_numpy()[mask])
        group_clfs[g] = clf
    for g in groups:
        mask = (test_df[gcol] == g).to_numpy()
        if not mask.any():
            continue
        acc = float((group_clfs[g].predict(X_test[mask]) == y_test[mask]).mean())
        conditional_rows[g] = {"n": int(mask.sum()), "accuracy": acc}
        conditional_avg.append(acc)
        conditional_worst.append(acc)

    return {
        "groups": groups,
        "gradient_audit": {
            "kappa_mix": structural["kappa_mix"],
            "kappa_ep": noisy["kappa_ep"],
            "E_shared": structural["E_shared"],
            "E_contrast": structural["E_contrast"],
            "sigma2": noisy["sigma2"],
            "n_items_per_group": noisy["n_per_condition"],
        },
        "capacity": {
            "shared": {
                "average_accuracy": shared_avg,
                "worst_group_accuracy": shared_worst,
                "per_group": shared_rows,
            },
            "group_conditioned": {
                "average_accuracy": float(np.mean(conditional_avg)),
                "worst_group_accuracy": float(np.min(conditional_worst)),
                "per_group": conditional_rows,
            },
        },
    }


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std()),
            "n": int(values.size)}


def run_seed(df, seed, axes):
    train_mask, test_mask, n_train, n_test = split_items(
        df, seed, TEST_FRACTION)
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, ngram_range=(1, 2), min_df=3,
        max_features=20000, strip_accents="unicode")
    vectorizer.fit(df.text[train_mask])
    X = vectorizer.transform(df.text)
    shared_clf = fit_logreg(X[train_mask], df.label.to_numpy()[train_mask])
    seed_result = {
        "seed": seed,
        "n_train_items": n_train,
        "n_test_items": n_test,
        "n_features": int(X.shape[1]),
        "shared_train_accuracy": float(
            shared_clf.score(X[train_mask], df.label.to_numpy()[train_mask])),
        "axes": {},
    }
    print(f"seed {seed}: items train/test={n_train}/{n_test}, "
          f"features={X.shape[1]}, shared train acc="
          f"{seed_result['shared_train_accuracy']:.4f}")
    for axis in axes:
        seed_result["axes"][axis] = audit_axis(
            df, X, train_mask, test_mask, axis, AXES[axis], shared_clf)
    return seed_result


def aggregate(seed_results, axes):
    out = {}
    for axis in axes:
        runs = [r["axes"][axis] for r in seed_results]
        audit_keys = ["kappa_mix", "kappa_ep", "E_shared", "E_contrast",
                      "sigma2"]
        out[axis] = {
            "groups": runs[0]["groups"],
            "gradient_audit": {
                key: _mean_std([r["gradient_audit"][key] for r in runs])
                for key in audit_keys
            },
            "capacity": {
                "shared": {
                    "average_accuracy": _mean_std(
                        [r["capacity"]["shared"]["average_accuracy"]
                         for r in runs]),
                    "worst_group_accuracy": _mean_std(
                        [r["capacity"]["shared"]["worst_group_accuracy"]
                         for r in runs]),
                },
                "group_conditioned": {
                    "average_accuracy": _mean_std(
                        [r["capacity"]["group_conditioned"]
                         ["average_accuracy"] for r in runs]),
                    "worst_group_accuracy": _mean_std(
                        [r["capacity"]["group_conditioned"]
                         ["worst_group_accuracy"] for r in runs]),
                },
            },
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--axes", nargs="+", default=["race", "age", "gender"])
    args = ap.parse_args()

    df, raw_rows = load(args.csv)
    print(f"rows={len(df)} (dropped Unsure: {raw_rows - len(df)}), "
          f"items={df.item_id.nunique()}, raters={df.rater_id.nunique()}")
    for axis in args.axes:
        if axis not in AXES:
            raise ValueError(f"unknown axis {axis}")

    seed_results = [run_seed(df, seed, args.axes) for seed in args.seeds]
    summary = aggregate(seed_results, args.axes)

    out = {
        "_metadata": measurement_metadata(
            "binary cross-entropy gradient of a TF-IDF logistic model",
            "equal_condition_weights", "supervised_split_by_item",
            "parameter_space_euclidean", "episode_noise_separate"),
        "dataset": {
            "name": "DICES-350",
            "rows_used": int(len(df)),
            "rows_dropped_unsure": int(raw_rows - len(df)),
            "items": int(df.item_id.nunique()),
            "raters": int(df.rater_id.nunique()),
            "label": "Q_overall: Yes (harmful) vs No; Unsure dropped",
            "test_fraction": TEST_FRACTION,
            "seeds": args.seeds,
        },
        "per_seed": seed_results,
        "summary": summary,
    }
    for axis, result in summary.items():
        a = result["gradient_audit"]
        cap = result["capacity"]
        print(f"\n[{axis}] groups={result['groups']}")
        print(f"  kappa_mix={a['kappa_mix']['mean']:.4f}+-{a['kappa_mix']['std']:.4f} "
              f"kappa_ep={a['kappa_ep']['mean']:.4f}+-{a['kappa_ep']['std']:.4f}")
        print(f"  E_shared={a['E_shared']['mean']:.5f} "
              f"E_contrast={a['E_contrast']['mean']:.5f} "
              f"sigma2={a['sigma2']['mean']:.5f}")
        print(f"  shared:      avg={cap['shared']['average_accuracy']['mean']:.4f}"
              f"+-{cap['shared']['average_accuracy']['std']:.4f} "
              f"worst={cap['shared']['worst_group_accuracy']['mean']:.4f}")
        print(f"  conditional: avg="
              f"{cap['group_conditioned']['average_accuracy']['mean']:.4f}"
              f"+-{cap['group_conditioned']['average_accuracy']['std']:.4f} "
              f"worst="
              f"{cap['group_conditioned']['worst_group_accuracy']['mean']:.4f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
