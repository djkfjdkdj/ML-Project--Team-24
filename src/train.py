"""Train and evaluate KNN, Neural Network and Random Forest.

Usage:  python -m src.train --data_dir data [--fast] [--out results]

Protocol: the DrivenData test set is unlabeled, so we hold out a stratified 20% of the
labelled training data and report micro-averaged F1 on it. Model selection (RF grid search,
NN restarts) uses only the 80% training portion. A DrivenData submission file is written
by the best model refit on all labelled data.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split

from .data import load_test, load_train
from .models import make_knn, make_nn, make_rf_search


def evaluate(name, model, Xtr, ytr, Xho, yho, fit_time):
    t = time.time(); p = model.predict(Xho); pred_time = time.time() - t
    sub = np.random.default_rng(0).choice(len(Xtr), min(20000, len(Xtr)), replace=False)
    ptr = model.predict(Xtr.iloc[sub]); ytr = ytr.iloc[sub]  # train score on a 20k sample (speed)
    return {
        "name": name,
        "f1_micro_train": f1_score(ytr, ptr, average="micro"),
        "f1_micro_holdout": f1_score(yho, p, average="micro"),
        "f1_macro_holdout": f1_score(yho, p, average="macro"),
        "fit_seconds": fit_time, "predict_seconds": pred_time,
        "confusion_matrix": confusion_matrix(yho, p).tolist(),
        "report": classification_report(yho, p, output_dict=True),
    }, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true", help="subsample 15%% of rows for a quick run")
    ap.add_argument("--knn_k", type=int, default=7)
    ap.add_argument("--nn_epochs", type=int, default=100)
    ap.add_argument("--nn_restarts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    X, y = load_train(a.data_dir)
    if a.fast:
        X, _, y, _ = train_test_split(X, y, train_size=0.15, stratify=y, random_state=a.seed)
    Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)
    print(f"train {Xtr.shape}  holdout {Xho.shape}")
    results, preds, models = [], {}, {}

    # ---- KNN ----
    print("KNN ..."); m = make_knn(Xtr, a.knn_k); t = time.time(); m.fit(Xtr, ytr)
    r, preds["KNN"] = evaluate("KNN", m, Xtr, ytr, Xho, yho, time.time() - t)
    r["params"] = {"n_neighbors": a.knn_k, "k_features": 20}; results.append(r); models["KNN"] = m

    # ---- Neural network (restarts chosen on an inner validation split) ----
    print("Neural network ...")
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.1, stratify=ytr, random_state=a.seed)
    t = time.time(); best_seed, best = 0, -1
    for s in range(a.nn_restarts):
        c = make_nn(Xa, seed=s, epochs=a.nn_epochs).fit(Xa, ya)
        sc = f1_score(yv, c.predict(Xv), average="micro"); print(f"  seed {s}: val F1 {sc:.4f}")
        if sc > best: best, best_seed = sc, s
    m = make_nn(Xtr, seed=best_seed, epochs=a.nn_epochs).fit(Xtr, ytr)
    r, preds["Neural Network"] = evaluate("Neural Network", m, Xtr, ytr, Xho, yho, time.time() - t)
    nn = m[-1]
    r["params"] = {"hidden": int(nn.W1_.shape[1]), "lr": nn.lr, "lambda": nn.lam,
                   "batch": nn.batch_size, "epochs": nn.epochs, "seed": best_seed}
    r["loss_history"] = nn.loss_history_; r["acc_history"] = nn.acc_history_
    results.append(r); models["Neural Network"] = m

    # ---- Random forest ----
    print("Random forest (grid search) ..."); gs = make_rf_search(Xtr, a.fast, a.seed)
    t = time.time(); gs.fit(Xtr, ytr)
    r, preds["Random Forest"] = evaluate("Random Forest", gs, Xtr, ytr, Xho, yho, time.time() - t)
    r["params"] = {k.split("__")[1]: v for k, v in gs.best_params_.items()}
    r["cv_f1_micro"] = gs.best_score_; results.append(r); models["Random Forest"] = gs.best_estimator_
    rf = gs.best_estimator_[-1]
    names = gs.best_estimator_[0].get_feature_names_out()
    imp = pd.Series(rf.feature_importances_, index=[n.split("__")[1] for n in names]).sort_values()

    # ---- artefacts ----
    for r in results:
        print(f"{r['name']:15s} train F1 {r['f1_micro_train']:.4f} | holdout F1 {r['f1_micro_holdout']:.4f}"
              f" | fit {r['fit_seconds']:.0f}s")
    json.dump({"n_train": len(Xtr), "n_holdout": len(Xho), "fast": a.fast, "results": results},
              open(out / "metrics.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(6, 3.4)); xs = np.arange(len(results)); w = 0.38
    ax.bar(xs - w/2, [r["f1_micro_train"] for r in results], w, label="train", color="#9bb7d4")
    ax.bar(xs + w/2, [r["f1_micro_holdout"] for r in results], w, label="holdout", color="#1f4e79")
    ax.set_xticks(xs); ax.set_xticklabels([r["name"] for r in results]); ax.set_ylim(0, 1)
    ax.set_ylabel("micro-averaged F1"); ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(out / "model_comparison.png", dpi=200); plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, r in zip(axs, results):
        ConfusionMatrixDisplay(np.array(r["confusion_matrix"]), display_labels=[1, 2, 3]).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d"); ax.set_title(r["name"])
    plt.tight_layout(); plt.savefig(out / "confusion_matrices.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(5, 3.4)); imp.tail(12).plot.barh(ax=ax, color="#1f4e79")
    ax.set_xlabel("Gini importance"); ax.set_title("Random Forest: top features")
    ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()
    plt.savefig(out / "rf_feature_importance.png", dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(5, 3.2)); ax.plot(results[1]["loss_history"], color="#c0392b")
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.set_title("Neural network training")
    ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()
    plt.savefig(out / "nn_loss.png", dpi=200); plt.close()

    # ---- DrivenData submission from best model, refit on all labelled data ----
    best_name = max(results, key=lambda r: r["f1_micro_holdout"])["name"]
    if (Path(a.data_dir) / "test_values.csv").exists():
        final = clone(models[best_name])
        final.fit(X, y)
        Xt = load_test(a.data_dir)
        pd.DataFrame({"building_id": Xt.index, "damage_grade": final.predict(Xt)}).to_csv(
            out / "submission.csv", index=False)
        print(f"submission.csv written using {best_name}")


if __name__ == "__main__":
    main()
