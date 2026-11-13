import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from collections import Counter


# =======================================================
# 1- data
# =======================================================
def load_csv_dataset(PATH):
    dataset = pd.read_csv(PATH)
    print("dataset loded")
    return dataset


import pandas as pd
import numpy as np


def analyze_reviews_csv(csv_path: str, sample_size: int = 2):
    """Fast and optimized analysis of a review dataset from a CSV file using Pandas."""
    line = "=" * 70
    print(f"\n{line}\nDATASET FIRST LOOK (OPTIMIZED FOR CSV)\n{line}")

    # ---------- 0) Read Data ----------
    try:
        # Read the file using the C engine for maximum speed
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # ---------- 1) Dataset Dimensions ----------
    print("\n[1] SHAPE AND COLUMNS")
    print(f"    - Total Rows: {len(df):,}")
    print(f"    - Total Columns: {len(df.columns)}")
    print("    - Columns Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"        * {col!r}: {dtype}")

    # ---------- 2) Null / Empty Values ----------
    print("\n[2] NULL / EMPTY VALUES")
    null_counts = df.isnull().sum()
    empty_counts = (df == "").sum()  # Count empty strings

    for col in df.columns:
        n_null = null_counts[col]
        n_empty = empty_counts[col]
        total_missing = n_null + n_empty
        pct = (total_missing / len(df)) * 100 if len(df) > 0 else 0
        status = "OK  " if total_missing == 0 else "WARN"
        print(f"    [{status}] {col}: {total_missing:,} missing ({pct:.2f}%)")

    # ---------- 3) Categorical Distributions ----------
    print("\n[3] CATEGORICAL DISTRIBUTIONS")

    if "rating" in df.columns:
        print("    - 'rating' distribution (rounded to nearest integer):")
        # Round decimals to nearest integer and then count
        rounded_ratings = df["rating"].round().astype(int)
        rating_counts = rounded_ratings.value_counts().sort_index()

        for rating, count in rating_counts.items():
            pct = (count / len(df)) * 100
            bar = "#" * int(pct / 2)
            print(f"        {rating} Star(s): {count:>7,} ({pct:5.1f}%) {bar}")

    if "is_buyer" in df.columns:
        print("    - 'is_buyer' distribution:")
        buyer_counts = df["is_buyer"].value_counts()
        for val, count in buyer_counts.items():
            pct = (count / len(df)) * 100
            print(f"        {val!r}: {count:,} ({pct:.1f}%)")

    # ---------- 4) Text Length Distribution ----------
    print("\n[4] TEXT LENGTH DISTRIBUTION (Characters)")
    text_columns = ["comment", "title"]

    for col in text_columns:
        if col in df.columns:
            # Calculate text length excluding nulls
            lengths = df[col].dropna().astype(str).str.len()
            if lengths.empty:
                continue

            print(f"    - {col!r}:")
            print(
                f"        * min: {lengths.min():,} | max: {lengths.max():,} | mean: {lengths.mean():.1f} | median: {lengths.median():.1f}"
            )

            # Bucket text lengths
            buckets = [0, 20, 50, 100, 200, 500, 1000, 2000, np.inf]
            labels = [
                "0-20",
                "20-50",
                "50-100",
                "100-200",
                "200-500",
                "500-1k",
                "1k-2k",
                "2k+",
            ]
            hist, _ = np.histogram(lengths, bins=buckets)
            total = hist.sum()

            for label, count in zip(labels, hist):
                if count > 0:
                    pct = (count / total) * 100
                    bar = "#" * int(pct / 2)
                    print(f"          {label:>7}: {count:>7,} ({pct:5.1f}%) {bar}")

    # ---------- 5) Reviews per Product (For Summarization) ----------
    if "product_id" in df.columns:
        print("\n[5] REVIEWS PER PRODUCT (For Summarization)")
        product_counts = df["product_id"].value_counts()

        print(f"    - Total unique products: {len(product_counts):,}")
        print(f"    - Avg reviews per product: {product_counts.mean():.1f}")
        print(f"    - Median reviews per product: {product_counts.median():.1f}")
        print(f"    - Max reviews for a single product: {product_counts.max():,}")

        # Number of products with enough reviews for summarization
        ge_5 = (product_counts >= 5).sum()
        ge_10 = (product_counts >= 10).sum()
        print(f"    - Products with >= 5 reviews: {ge_5:,}")
        print(f"    - Products with >= 10 reviews: {ge_10:,}")

    # ---------- 6) Duplicates ----------
    print("\n[6] DUPLICATES")
    if "comment" in df.columns:
        n_duplicates = df["comment"].duplicated().sum()
        pct = (n_duplicates / len(df)) * 100
        print(f"    - Duplicate comments: {n_duplicates:,} ({pct:.2f}%)")

    # ---------- 7) Samples ----------
    print("\n[7] SAMPLES")
    for i in range(min(sample_size, len(df))):
        print(f"\n    --- Sample {i + 1} ---")
        for col in df.columns:
            val = str(df.iloc[i][col]).replace("\n", " ")
            if len(val) > 100:
                val = val[:100] + "..."
            print(f"        [{col}] {val!r}")

    print(f"\n{line}\nAnalysis finished.\n{line}")


# =======================================================
# 2- Pre Process
# =======================================================
import pandas as pd
import os


def preprocess_reviews(csv_path, min_len=20, max_len=1000):
    output_path = "data/processed_datasets/digikala_comments_cleaned_min20_max1000.csv"

    # 1. Check if the cleaned file already exists
    if os.path.exists(output_path):
        print("=" * 60)
        print("Existing cleaned dataset found!")
        print(f"Loading data from: {output_path}")
        print("=" * 60)
        return pd.read_csv(output_path)

    # If file doesn't exist, start preprocessing
    print("=" * 60)
    print("Starting preprocessing of the Digikala reviews dataset...")
    print("=" * 60)

    # 2. Read only the two required columns
    print("\n[1] Reading the CSV file...")
    df = pd.read_csv(csv_path, usecols=["product_id", "comment"])
    initial_rows = len(df)
    print(f"    - Total initial rows: {initial_rows:,}")

    # 3. Remove empty or null values
    df = df.dropna(subset=["comment"])
    df = df[df["comment"].str.strip() != ""]
    print(f"    - Row count after removing empty values: {len(df):,}")

    # 4. Remove duplicate comments
    print("\n[2] Removing duplicate comments...")
    df = df.drop_duplicates(subset=["comment"])
    print(f"    - Row count after removing duplicates: {len(df):,}")

    # 5. Filter based on text length
    print(
        f"\n[3] Applying text length filters (min {min_len} and max {max_len} characters)..."
    )
    comment_lengths = df["comment"].str.len()

    df_cleaned = df[(comment_lengths >= min_len) & (comment_lengths <= max_len)].copy()

    short_removed = (comment_lengths < min_len).sum()
    long_removed = (comment_lengths > max_len).sum()

    print(f"    - Removed comments shorter than {min_len} characters: {short_removed:,}")
    print(f"    - Removed comments longer than {max_len} characters: {long_removed:,}")
    print(f"    - Final number of clean rows: {len(df_cleaned):,}")

    # 6. Reset the index
    df_cleaned = df_cleaned.reset_index(drop=True)

    # 7. Save the cleaned dataset
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n[4] Saving the cleaned dataset to {output_path}...")
    df_cleaned.to_csv(output_path, index=False)

    print("=" * 60)
    print("Preprocessing completed successfully! Data is ready for the next step.")
    print("=" * 60)

    return df_cleaned


# =======================================================
# 3- Smart BATCH
# =======================================================


if __name__ == "__main__":
    DATASET_PATH = "data/raw_datasets/digikala_comments.csv"

    raw_datset = load_csv_dataset(DATASET_PATH)
    # print(raw_datset.head())
    # analyze_reviews_csv(DATASET_PATH)

    preprocess_reviews(DATASET_PATH)
    analyze_reviews_csv("data/processed_datasets/digikala_comments_cleaned_min20_max1000.csv")
