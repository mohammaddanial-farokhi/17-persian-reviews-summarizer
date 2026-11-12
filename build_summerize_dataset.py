import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from collections import Counter

# =======================================================
# 1- data
# =======================================================
def load_csv_dataset(PATH):
    dataset=pd.read_csv(PATH)
    print("dataset loded")
    return dataset

def analyze_reviews_csv(csv_path: str, sample_size: int = 2):
    """تحلیل سریع و بهینه یک دیتاست نظرات از فایل CSV با استفاده از Pandas"""
    line = "=" * 70
    print(f"\n{line}\nDATASET FIRST LOOK (OPTIMIZED FOR CSV)\n{line}")

    # ---------- 0) خواندن داده ----------
    try:
        # خواندن فایل با انجین C برای بالاترین سرعت
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"خطا در خواندن فایل: {e}")
        return

    # ---------- 1) ابعاد دیتاست ----------
    print("\n[1] SHAPE AND COLUMNS")
    print(f"    - Total Rows: {len(df):,}")
    print(f"    - Total Columns: {len(df.columns)}")
    print("    - Columns Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"        * {col!r}: {dtype}")

    # ---------- 2) مقادیر خالی (Null / Empty) ----------
    print("\n[2] NULL / EMPTY VALUES")
    null_counts = df.isnull().sum()
    empty_counts = (df == "").sum() # شمارش رشته‌های خالی
    
    for col in df.columns:
        n_null = null_counts[col]
        n_empty = empty_counts[col]
        total_missing = n_null + n_empty
        pct = (total_missing / len(df)) * 100 if len(df) > 0 else 0
        status = "OK  " if total_missing == 0 else "WARN"
        print(f"    [{status}] {col}: {total_missing:,} missing ({pct:.2f}%)")

    # ---------- 3) توزیع ستون‌های دسته‌بندی شده ----------
    print("\n[3] CATEGORICAL DISTRIBUTIONS")
    
    if 'rating' in df.columns:
        print("    - 'rating' distribution:")
        rating_counts = df['rating'].value_counts().sort_index()
        for rating, count in rating_counts.items():
            pct = (count / len(df)) * 100
            bar = "#" * int(pct / 2)
            print(f"        {rating} Star(s): {count:>7,} ({pct:5.1f}%) {bar}")

    if 'is_buyer' in df.columns:
        print("    - 'is_buyer' distribution:")
        buyer_counts = df['is_buyer'].value_counts()
        for val, count in buyer_counts.items():
            pct = (count / len(df)) * 100
            print(f"        {val!r}: {count:,} ({pct:.1f}%)")

    # ---------- 4) توزیع طول متن ----------
    print("\n[4] TEXT LENGTH DISTRIBUTION (Characters)")
    text_columns = ['comment', 'title']
    
    for col in text_columns:
        if col in df.columns:
            # محاسبه طول متن با حذف مقادیر تهی
            lengths = df[col].dropna().astype(str).str.len()
            if lengths.empty:
                continue
                
            print(f"    - {col!r}:")
            print(f"        * min: {lengths.min():,} | max: {lengths.max():,} | mean: {lengths.mean():.1f} | median: {lengths.median():.1f}")
            
            # تقسیم‌بندی طول متن
            buckets = [0, 20, 50, 100, 200, 500, 1000, 2000, np.inf]
            labels = ["0-20", "20-50", "50-100", "100-200", "200-500", "500-1k", "1k-2k", "2k+"]
            hist, _ = np.histogram(lengths, bins=buckets)
            total = hist.sum()
            
            for label, count in zip(labels, hist):
                if count > 0:
                    pct = (count / total) * 100
                    bar = "#" * int(pct / 2)
                    print(f"          {label:>7}: {count:>7,} ({pct:5.1f}%) {bar}")

    # ---------- 5) تحلیل محصولات (برای خلاصه‌سازی) ----------
    if 'product_id' in df.columns:
        print("\n[5] REVIEWS PER PRODUCT (For Summarization)")
        product_counts = df['product_id'].value_counts()
        
        print(f"    - Total unique products: {len(product_counts):,}")
        print(f"    - Avg reviews per product: {product_counts.mean():.1f}")
        print(f"    - Median reviews per product: {product_counts.median():.1f}")
        print(f"    - Max reviews for a single product: {product_counts.max():,}")
        
        # تعداد محصولاتی که نظر کافی برای خلاصه‌سازی دارند
        ge_5 = (product_counts >= 5).sum()
        ge_10 = (product_counts >= 10).sum()
        print(f"    - Products with >= 5 reviews: {ge_5:,}")
        print(f"    - Products with >= 10 reviews: {ge_10:,}")

    # ---------- 6) داده‌های تکراری ----------
    print("\n[6] DUPLICATES")
    if 'comment' in df.columns:
        n_duplicates = df['comment'].duplicated().sum()
        pct = (n_duplicates / len(df)) * 100
        print(f"    - Duplicate comments: {n_duplicates:,} ({pct:.2f}%)")

    # ---------- 7) نمایش نمونه ----------
    print("\n[7] SAMPLES")
    for i in range(min(sample_size, len(df))):
        print(f"\n    --- Sample {i + 1} ---")
        for col in df.columns:
            val = str(df.iloc[i][col]).replace("\n", " ")
            if len(val) > 100:
                val = val[:100] + "..."
            print(f"        [{col}] {val!r}")

    print(f"\n{line}\nAnalysis finished.\n{line}")




if __name__=="__main__":
    DATASETPATH="data/datasets/digikala_comments.csv"

    raw_datset=load_csv_dataset(DATASETPATH)
    # print(raw_datset.head())
    
    analyze_reviews_csv(DATASETPATH)