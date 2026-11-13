import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from collections import Counter
import json
import os
import re
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm


# =======================================================
# 1- data
# =======================================================
def load_csv_dataset(PATH):
    dataset = pd.read_csv(PATH)
    print("dataset loded")
    return dataset


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
def smart_batch_reviews(
    csv_path, max_chars_per_batch=2000, min_chars_per_batch=500, min_comments_per_product=10
):
    """
    Groups reviews by product_id into smart batches.
    Filters products with minimum comments and drops batches that are too short.
    """

    output_path = "data/processed_datasets/batches_for_summarization.jsonl"

    # 1. Check if the file already exists (Caching)
    if os.path.exists(output_path):
        print("=" * 60)
        print("Existing batches file found!")
        print(f"Loading data from: {output_path}")
        print("=" * 60)

        batches = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                batches.append(json.loads(line))
        return batches

    # 2. Start batching
    print("=" * 60)
    print(
        f"Starting smart batching (Max {max_chars_per_batch} chars, Min {min_chars_per_batch} chars)..."
    )
    print(f"Min comments per product: {min_comments_per_product}")
    print("=" * 60)

    # Read the cleaned dataset
    df = pd.read_csv(csv_path)

    # 3. Filter products that have at least `min_comments_per_product` comments
    print("\n[1] Filtering products with few comments...")
    initial_products = df["product_id"].nunique()
    df = df[df.groupby("product_id")["comment"].transform("count") >= min_comments_per_product]
    filtered_products = df["product_id"].nunique()
    print(f"    - Products before filter: {initial_products:,}")
    print(f"    - Products after filter: {filtered_products:,}")
    print(f"    - Total comments to process: {len(df):,}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    batches = []
    batch_id = 0

    # 4. Group by product_id
    print("\n[2] Creating batches and formatting text...")
    for product_id, group in df.groupby("product_id"):
        current_comments = []
        current_chars = 0

        for comment in group["comment"]:
            # Clean internal newlines and extra spaces
            comment_str = str(comment).strip()
            comment_str = re.sub(r"[\r\n]+", " ", comment_str)
            comment_str = re.sub(r"\s+", " ", comment_str)

            comment_len = len(comment_str)

            if current_chars + comment_len > max_chars_per_batch and current_comments:
                # Only save if the batch meets the minimum character requirement
                if current_chars >= min_chars_per_batch:
                    formatted_text = "\n".join(
                        [f"<<< نظر {i+1} >>>\n{c}" for i, c in enumerate(current_comments)]
                    )
                    batches.append(
                        {
                            "batch_id": batch_id,
                            "product_id": int(product_id),
                            "comments_list": current_comments,
                            "formatted_text": formatted_text,
                            "total_chars": current_chars,
                            "num_comments": len(current_comments),
                        }
                    )
                    batch_id += 1

                # Start new batch
                current_comments = [comment_str]
                current_chars = comment_len
            else:
                current_comments.append(comment_str)
                current_chars += comment_len

        # Save the last batch for this product
        if current_comments and current_chars >= min_chars_per_batch:
            formatted_text = "\n".join(
                [f"<<< نظر {i+1} >>>\n{c}" for i, c in enumerate(current_comments)]
            )
            batches.append(
                {
                    "batch_id": batch_id,
                    "product_id": int(product_id),
                    "comments_list": current_comments,
                    "formatted_text": formatted_text,
                    "total_chars": current_chars,
                    "num_comments": len(current_comments),
                }
            )
            batch_id += 1

    # 5. Save to JSONL
    print(f"\n[3] Saving {len(batches):,} batches to '{output_path}'...")
    with open(output_path, "w", encoding="utf-8") as f:
        for batch in batches:
            f.write(json.dumps(batch, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("Smart batching completed successfully! Data is ready.")
    print("=" * 60)

    # Print a sample batch for preview
    print("\nSample Batch Preview:")
    if batches:
        print("Formatted Text:")
        print(batches[0]["formatted_text"][:600] + "\n...")

    return batches


def analyze_batches_jsonl(file_path):
    """
    Validates and analyzes the batched JSONL dataset before feeding it to the LLM.
    """
    line = "=" * 70
    print(f"\n{line}\nBATCHES DATASET ANALYSIS\n{line}")

    # ---------- 0. Load Data ----------
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    try:
        # Read JSONL file into a Pandas DataFrame
        df = pd.read_json(file_path, lines=True)
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return

    # ---------- 1. Basic Structure ----------
    print("\n[1] SHAPE AND COLUMNS")
    print(f"    - Total Batches: {len(df):,}")
    print(f"    - Total Columns: {len(df.columns)}")
    for col in df.columns:
        print(f"        * {col!r}: {df[col].dtype}")

    # ---------- 2. ID Validations ----------
    print("\n[2] ID VALIDATIONS")
    # Check batch_id uniqueness
    is_batch_unique = df["batch_id"].is_unique
    print(f"    - Are all 'batch_id's unique? {'YES' if is_batch_unique else 'NO'}")
    if not is_batch_unique:
        print(f"        * Duplicate batch_ids found: {df['batch_id'].duplicated().sum():,}")

    # Check product_id uniqueness and distribution
    unique_products = df["product_id"].nunique()
    print(f"    - Total unique 'product_id's: {unique_products:,}")
    print(f"    - Avg batches per product: {len(df) / unique_products:.1f}")

    # Show products with the most batches (just for insight)
    top_products = df["product_id"].value_counts().head(3)
    print("    - Top 3 products with the most batches:")
    for pid, count in top_products.items():
        print(f"        * Product {pid}: {count} batches")

    # ---------- 3. Empty / Null Checks ----------
    print("\n[3] EMPTY / NULL CHECKS")

    # Check if comments_list is empty
    def is_list_empty(val):
        return not isinstance(val, list) or len(val) == 0

    empty_lists = df["comments_list"].apply(is_list_empty).sum()
    print(f"    - Batches with empty 'comments_list': {empty_lists:,}")

    # Check if formatted_text is empty
    empty_texts = (
        df["formatted_text"].isnull() | (df["formatted_text"].str.strip() == "")
    ).sum()
    print(f"    - Batches with empty 'formatted_text': {empty_texts:,}")

    # ---------- 4. Statistics: total_chars ----------
    print("\n[4] STATISTICS: 'total_chars'")
    print(
        f"    - min: {df['total_chars'].min():,} | max: {df['total_chars'].max():,} | mean: {df['total_chars'].mean():.1f} | median: {df['total_chars'].median():.1f}"
    )

    # Buckets for total_chars (based on min 500 and max 2000)
    buckets = [0, 500, 750, 1000, 1250, 1500, 1750, 2000, np.inf]
    labels = [
        "<500",
        "500-750",
        "750-1k",
        "1k-1.25k",
        "1.25k-1.5k",
        "1.5k-1.75k",
        "1.75k-2k",
        ">2k",
    ]
    hist, _ = np.histogram(df["total_chars"], bins=buckets)
    total = hist.sum()

    for label, count in zip(labels, hist):
        if count > 0:
            pct = (count / total) * 100
            bar = "#" * int(pct / 2)
            print(f"        {label:>10}: {count:>7,} ({pct:5.1f}%) {bar}")

    # ---------- 5. Statistics: num_comments ----------
    print("\n[5] STATISTICS: 'num_comments'")
    print(
        f"    - min: {df['num_comments'].min():,} | max: {df['num_comments'].max():,} | mean: {df['num_comments'].mean():.1f} | median: {df['num_comments'].median():.1f}"
    )

    # Buckets for num_comments (based on min 10 comments per product)
    comment_buckets = [0, 5, 10, 15, 20, 30, 50, np.inf]
    comment_labels = ["<5", "5-10", "10-15", "15-20", "20-30", "30-50", "50+"]
    hist_c, _ = np.histogram(df["num_comments"], bins=comment_buckets)
    total_c = hist_c.sum()

    for label, count in zip(comment_labels, hist_c):
        if count > 0:
            pct = (count / total_c) * 100
            bar = "#" * int(pct / 2)
            print(f"        {label:>10}: {count:>7,} ({pct:5.1f}%) {bar}")

    print(f"\n{line}\nAnalysis finished.\n{line}")


# =======================================================
# 4- load model & tokenizer
# =======================================================
def load_model_for_inference(model_name, max_seq_length=4096):
    """
    Loads model using Unsloth for fast inference (generating summaries).
    """
    print(f"Loading model with Unsloth for Inference: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,  # Increased to 4096 to handle long inputs
        dtype=torch.bfloat16,
        load_in_4bit=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    FastLanguageModel.for_inference(model)

    print("✅ Model loaded with Unsloth for Inference!")
    return model, tokenizer


# =======================================================
# 5- Prompt Engineering & Generation
# =======================================================
def generate_summaries(model, tokenizer, test_mode=False, batch_size=6):
    """
    Generate Persian summaries using Batch Inference for maximum GPU utilization.
    """
    INPUT_FILE = "data/processed_datasets/batches_for_summarization.jsonl"

    OUTPUT_FILE = (
        "data/processed_datasets/synthetic_summary_test.jsonl"
        if test_mode
        else "data/processed_datasets/synthetic_summary_dataset.jsonl"
    )

    SYSTEM_PROMPT = """تو یک متخصص تحلیل و خلاصه‌سازی نظرات مشتریان هستی.

مجموعه‌ای از نظرات کاربران درباره یک محصول به تو داده می‌شود. هر نظر با تگ‌هایی مانند <<< نظر 1 >>> از سایر نظرات جدا شده است. این تگ‌ها صرفاً برای جداسازی هستند و نباید در خروجی ظاهر شوند.

وظیفه تو تولید یک خلاصه یکپارچه، روان و بی‌طرفانه از مجموع نظرات است.

در خلاصه خود:
1. مهم‌ترین نقاط قوت و مزایای محصول را بیان کن.
2. مهم‌ترین نقاط ضعف و مشکلات گزارش‌شده را مشخص کن.
3. به موضوعات پرتکرار توجه بیشتری داشته باش، اما مشکلات مهم و قابل‌توجه را حتی اگر فقط در یک نظر مطرح شده‌اند نادیده نگیر. در عین حال، میزان فراگیری هر مشکل را به‌درستی منعکس کن و از تعمیم تجربه شخصی یک کاربر به کل محصول خودداری کن. مشکلات ناشی از استفاده نادرست، حوادث شخصی یا آسیب‌هایی که کاربر خود به محصول وارد کرده است، نباید به‌عنوان ضعف ذاتی محصول معرفی شوند.
4. در تشخیص نقاط قوت و ضعف، قطبیت هر ویژگی را با دقت حفظ کن. یک ویژگی مثبت را به‌عنوان عیب یا یک ویژگی منفی را به‌عنوان مزیت معرفی نکن.
5. صرف اشاره به یک ویژگی، مشخصات یا واقعیت درباره محصول، به معنای رضایت یا نارضایتی کاربر از آن نیست. فقط زمانی موضوعی را به‌عنوان نقطه قوت یا ضعف مطرح کن که از متن نظر، رضایت یا نارضایتی کاربر نسبت به آن قابل برداشت باشد.
6. از ارائه درصد، نسبت یا آمار عددی درباره میزان رضایت کاربران خودداری کن، مگر اینکه این مقادیر به‌صورت صریح در نظرات ذکر شده باشند. تعداد یا نسبت کاربران را از روی نظرات به‌صورت تخمینی محاسبه یا اعلام نکن.
7. در صورت وجود اختلاف نظر میان کاربران، آن را حفظ کن.
8. حس کلی کاربران نسبت به محصول را بر اساس مجموع نظرات بیان کن.

فقط از اطلاعات موجود در نظرات استفاده کن و هیچ اطلاعاتی از دانش قبلی خود اضافه نکن. همچنین از تکرار مطالب و نقل‌قول مستقیم (کپی کردن عین جملات) از کاربران خودداری کن.

خروجی باید به زبان فارسی روان و رسمی، در حداکثر 4 تا 5 جمله باشد و فقط شامل متن خلاصه شود؛ بدون عنوان، مقدمه یا توضیحات اضافی."""

    # 1. Load already processed batch IDs
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data["batch_id"])
                except (json.JSONDecodeError, KeyError):
                    print("Warning: Invalid record found in output file.")
        print(f"Found {len(processed_ids)} processed batches. Resuming...")

    # 2. Load input batches
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_batches = [json.loads(line) for line in f if line.strip()]

    # 3. Filter unprocessed batches
    batches_to_process = [
        batch for batch in all_batches if batch["batch_id"] not in processed_ids
    ]

    if test_mode:
        batches_to_process = batches_to_process[:5]
        print("TEST MODE: Processing up to 5 batches.")

    if not batches_to_process:
        print("All batches have been processed already!")
        return

    print(
        f"Starting inference on {len(batches_to_process)} batches with batch_size={batch_size}..."
    )

    # 4. Model and Tokenizer configuration for Batch Inference
    model.eval()

    # CRITICAL: Decoder-only models (like Gemma) MUST use left padding for batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 5. Process in Chunks
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:

        # Iterate over the data in steps of batch_size
        for i in tqdm(
            range(0, len(batches_to_process), batch_size), desc="Generating Summaries"
        ):
            chunk = batches_to_process[i : i + batch_size]

            prompts = []
            for batch in chunk:
                user_content = (
                    f"{SYSTEM_PROMPT}\n\nنظرات کاربران:\n{batch['formatted_text']}\n"
                )
                messages = [{"role": "user", "content": user_content}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

            # Tokenize the whole batch at once (with padding)
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536,  # Prevent VRAM overflow
            ).to(model.device)

            # Generate for the whole batch
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode the whole batch
            generated_texts = tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1] :],
                skip_special_tokens=True,
            )

            # Save results
            for batch, generated_text in zip(chunk, generated_texts):
                generated_text = generated_text.strip()
                if not generated_text:
                    print(f"Warning: Empty summary for batch {batch['batch_id']}")
                    continue

                result_record = {
                    "batch_id": batch["batch_id"],
                    "product_id": batch["product_id"],
                    "num_comments": batch["num_comments"],
                    "input_text": batch["formatted_text"],
                    "summary": generated_text,
                }
                f_out.write(json.dumps(result_record, ensure_ascii=False) + "\n")

            # Flush after each chunk to save memory and prevent data loss
            f_out.flush()

    print("=" * 60)
    print(f"Inference completed! Results saved to {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    DATASET_PATH = "data/raw_datasets/digikala_comments.csv"
    CLEAN_DATASET = "data/processed_datasets/digikala_comments_cleaned_min20_max1000.csv"
    MODEL_NAME = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"

    raw_datset = load_csv_dataset(DATASET_PATH)
    # print(raw_datset.head())
    # analyze_reviews_csv(DATASET_PATH)

    preprocess_reviews(DATASET_PATH)
    # analyze_reviews_csv(CLEAN_DATASET)

    smart_batch_reviews(CLEAN_DATASET)
    # analyze_batches_jsonl("data/processed_datasets/batches_for_summarization.jsonl")

    base_model, base_tokenizer = load_model_for_inference(MODEL_NAME)

    generate_summaries(base_model, base_tokenizer)
