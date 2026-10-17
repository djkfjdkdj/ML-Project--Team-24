"""Model factories: KNN, Neural Network, Random Forest."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

from .data import onehot_preprocessor, ordinal_preprocessor
from .neural_net import NeuralNetClassifier


def make_knn(X, n_neighbors=7, k_features=20):
    prep, scaler = onehot_preprocessor(X)
    return make_pipeline(prep, scaler, SelectKBest(f_classif, k=k_features),
                         KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1))


def make_nn(X, seed=0, epochs=100, lr=2.0, lam=1e-4, batch_size=1000):
    prep, scaler = onehot_preprocessor(X)
    return make_pipeline(prep, scaler, NeuralNetClassifier(
        lr=lr, lam=lam, batch_size=batch_size, epochs=epochs, seed=seed))


def make_rf_search(X, fast=False, seed=0):
    pipe = make_pipeline(ordinal_preprocessor(X),
                         RandomForestClassifier(random_state=seed, n_jobs=-1))
    grid = {"randomforestclassifier__min_samples_leaf": [1, 5],
            "randomforestclassifier__n_estimators": [50] if fast else [50, 100]}
    return GridSearchCV(pipe, grid, scoring="f1_micro", cv=3, refit=True)


if __name__ == "__main__":
    from .data import load_train

    print("Building model pipelines with a sample of training data...")
    X_sample, y_sample = load_train("data")
    X_sample = X_sample.head(100)
    y_sample = y_sample.head(100)

    knn_pipe = make_knn(X_sample)
    print("\n1. KNN Pipeline Steps:")
    for name, step in knn_pipe.steps:
        print(f"   - {name}: {step.__class__.__name__}")

    nn_pipe = make_nn(X_sample)
    print("\n2. Neural Network Pipeline Steps:")
    for name, step in nn_pipe.steps:
        print(f"   - {name}: {step.__class__.__name__}")

    rf_search = make_rf_search(X_sample, fast=True)
    print("\n3. Random Forest Search Pipeline Steps:")
    for name, step in rf_search.estimator.steps:
        print(f"   - {name}: {step.__class__.__name__}")
    print(f"   Hyperparameter grid: {rf_search.param_grid}")

