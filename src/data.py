"""Data loading and preprocessing for the Richter's Predictor dataset."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "damage_grade"
ID = "building_id"


def load_train(data_dir):
    d = Path(data_dir)
    X = pd.read_csv(d / "train_values.csv", index_col=ID)
    y = pd.read_csv(d / "train_labels.csv", index_col=ID)[TARGET]
    return X, y.loc[X.index]


def load_test(data_dir):
    return pd.read_csv(Path(data_dir) / "test_values.csv", index_col=ID)


def split_columns(X):
    cat = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return num, cat


def onehot_preprocessor(X):
    """One-hot categoricals, standardise everything (used by KNN and the NN)."""
    num, cat = split_columns(X)
    ct = ColumnTransformer(
        [("num", "passthrough", num),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return ct, StandardScaler()


def ordinal_preprocessor(X):
    """Ordinal-encode categoricals (trees do not need scaling)."""
    num, cat = split_columns(X)
    return ColumnTransformer(
        [("num", "passthrough", num),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat)])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test and inspect data loading")
    parser.add_argument("--data_dir", default="data", help="Path to data folder")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not (data_path / "train_values.csv").exists() and Path("data_synth").exists():
        data_path = Path("data_synth")

    print(f"Loading data from: {data_path}")
    X_train, y_train = load_train(data_path)
    num_cols, cat_cols = split_columns(X_train)

    print(f"Train features (X): {X_train.shape[0]:,} rows, {X_train.shape[1]} columns")
    print(f"Train targets  (y): {y_train.shape[0]:,} rows")
    print(f"Numerical features:   {len(num_cols)} -> {num_cols[:5]}...")
    print(f"Categorical features: {len(cat_cols)} -> {cat_cols}")
    print("\nClass distribution (damage_grade):")
    print(y_train.value_counts(normalize=True).mul(100).round(2).to_string())

    if (data_path / "test_values.csv").exists():
        X_test = load_test(data_path)
        print(f"\nTest features (unlabeled): {X_test.shape[0]:,} rows, {X_test.shape[1]} columns")

