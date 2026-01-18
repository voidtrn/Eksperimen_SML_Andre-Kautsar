#!/usr/bin/env python3
"""
Run (local):
  python automate_Andre-Kautsar.py --input ../student-performance_raw.csv
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="student-performance_raw.csv", help="Path to raw CSV")
    p.add_argument("--outdir", default="student-performance_preprocessing", help="Output directory")
    p.add_argument("--target", default="G3", help="Target column")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {in_path.resolve()}")

    df = pd.read_csv(in_path)
    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[args.target])
    y = df[args.target]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    scaler = StandardScaler()
    X_train_scaled = X_train_enc.copy()
    X_test_scaled = X_test_enc.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train_enc[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test_enc[num_cols])

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train_scaled.copy()
    train_df[args.target] = y_train.values
    test_df = X_test_scaled.copy()
    test_df[args.target] = y_test.values

    train_df.to_csv(out_dir / "train_processed.csv", index=False)
    test_df.to_csv(out_dir / "test_processed.csv", index=False)

    print("Preprocessing done")

if __name__ == "__main__":
    main()