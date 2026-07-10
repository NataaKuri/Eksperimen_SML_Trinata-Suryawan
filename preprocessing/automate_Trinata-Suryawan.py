"""
automate_Trinata-Suryawan.py
=============================
Otomatisasi preprocessing dataset Heart Disease (Kriteria 1 - Skilled/Advance).

Merupakan konversi langkah-langkah preprocessing manual pada notebook
`Eksperimen_Trinata-Suryawan.ipynb` menjadi fungsi yang dapat dijalankan
ulang secara konsisten. Menghasilkan data yang siap dilatih (train.csv & test.csv).

Cara pakai:
    python automate_Trinata-Suryawan.py \
        --input heart_raw.csv \
        --output-dir namadataset_preprocessing

Jika dijalankan tanpa argumen, akan menggunakan nilai default di atas.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
NUM_FEATURES = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari file CSV."""
    df = pd.read_csv(path)
    print(f"[load_data] Data dimuat: {df.shape}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Menangani missing value terselubung (nilai 0 yang tidak masuk akal)."""
    df = df.copy()
    for col in ["Cholesterol", "RestingBP"]:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    print(f"[handle_missing] Sisa missing value: {int(df.isnull().sum().sum())}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Menghapus baris duplikat."""
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[remove_duplicates] Duplikat dihapus: {before - df.shape[0]}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encoding fitur kategorikal (binary mapping + one-hot)."""
    df = df.copy()
    df["Sex"] = df["Sex"].map({"M": 1, "F": 0})
    df["ExerciseAngina"] = df["ExerciseAngina"].map({"Y": 1, "N": 0})

    multi_cat = ["ChestPainType", "RestingECG", "ST_Slope"]
    df = pd.get_dummies(df, columns=multi_cat, drop_first=True)

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"[encode_features] Shape setelah encoding: {df.shape}")
    return df


def split_and_scale(df: pd.DataFrame, target: str = "HeartDisease"):
    """Split train-test lalu standardisasi fitur numerik."""
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUM_FEATURES] = scaler.fit_transform(X_train[NUM_FEATURES])
    X_test[NUM_FEATURES] = scaler.transform(X_test[NUM_FEATURES])

    train_out = X_train.copy()
    train_out[target] = y_train.values
    test_out = X_test.copy()
    test_out[target] = y_test.values

    print(f"[split_and_scale] Train: {train_out.shape} | Test: {test_out.shape}")
    return train_out, test_out


def preprocess(input_path: str, output_dir: str):
    """Menjalankan seluruh pipeline preprocessing dan menyimpan hasilnya."""
    df = load_data(input_path)
    df = handle_missing(df)
    df = remove_duplicates(df)
    df = encode_features(df)
    train_out, test_out = split_and_scale(df)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"[preprocess] Selesai. Disimpan ke:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    return train_out, test_out


def main():
    parser = argparse.ArgumentParser(description="Otomatisasi preprocessing Heart Disease.")
    parser.add_argument("--input", default="heart_raw.csv", help="Path file CSV mentah.")
    parser.add_argument("--output-dir", default="namadataset_preprocessing",
                        help="Folder output hasil preprocessing.")
    args = parser.parse_args()
    preprocess(args.input, args.output_dir)


if __name__ == "__main__":
    main()
