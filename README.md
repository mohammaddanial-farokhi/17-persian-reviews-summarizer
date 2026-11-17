# Persian Customer Reviews Summarizer (Gemma-3-4B)

![Python](https://img.shields.io/badge/Python-3.12.16-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.x-yellow.svg)
![Unsloth](https://img.shields.io/badge/Unsloth-Fast_Inference-orange.svg)
![Task](https://img.shields.io/badge/Task-Text%20Summarization-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## 🧠 About The Project

A complete end-to-end pipeline to build a high-quality **synthetic dataset for Persian text summarization**. 

This project processes raw customer reviews from Digikala (an Iranian e-commerce platform), intelligently groups them by product, and utilizes the **Gemma-3-4B** language model (via **Unsloth** for accelerated inference) to generate human-like, unbiased summaries for each product's reviews. 

The pipeline features robust data cleaning, smart text batching, and highly optimized batch inference to generate thousands of high-quality summary-target pairs.

---

## 🔒 Dataset Access (Important)

The final generated dataset (containing 4,000+ high-quality reviews-to-summary pairs) is available in this repository as **`dataset.zip`**. 

**⚠️ Security Notice:** 
The `dataset.zip` file is **password-protected**. 
To access the dataset, please send an email to **`mdanialfarokhi@gmail.com`** to request the password.

---

## 🎯 Key Features

1. **Robust Data Preprocessing:** Automatically cleans raw reviews, removes duplicates, handles nulls, and filters texts based on optimal character lengths (min 20, max 1000).
2. **Smart Batching:** Groups reviews by `product_id` and dynamically creates text batches (between 500 to 2000 characters) to ensure the LLM gets enough context without exceeding its limits.
3. **Fast Batch Inference:** Utilizes Unsloth's `FastLanguageModel` with `padding_side="left"` to generate summaries for multiple products simultaneously, maximizing GPU utilization.
4. **Resumable Generation:** Automatically detects if the process was interrupted, reads the generated JSONL file, and resumes from the last processed batch without losing progress.
5. **Expert Prompt Engineering:** Uses a meticulously crafted Persian system prompt to ensure the summaries are unbiased, capture both pros and cons, and strictly avoid hallucination or personal user biases.

---

## 📂 Project Structure

```text
.
├── build_summerize_dataset.py             # Main pipeline script
├── dataset.zip                            # Password-protected final dataset
├── README.md
└── requirements.txt
```

---

## ⚙️ Generation Configuration

| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Base Model** | `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` | Pre-quantized 4-bit model for fast inference |
| **Max Seq Length** | 4096 | To handle long batches of reviews |
| **Batch Size** | 6 | For parallel GPU inference |
| **Temperature** | 0.3 | Lower temperature for factual and consistent summaries |
| **Top_p** | 0.9 | Nucleus sampling |
| **Max New Tokens** | 256 | Maximum length of the generated summary |

---

## 🚀 Usage

### 1. Install Dependencies
Ensure you have Python 3.12+ and CUDA installed. Then install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
To start the entire process (Data Cleaning -> Smart Batching -> LLM Generation):
```bash
python build_summerize_dataset.py
```
*(Note: The script will automatically skip steps if their output files already exist, allowing you to resume safely if the process is interrupted.)*

---

## ⚠️ Known Limitations

*   **Model Bias:** While the prompt is designed to minimize bias, the generated summaries are based on the collective sentiment of the reviews, which might not always reflect the absolute truth about the product.
*   **Hardware Requirement:** This pipeline requires a CUDA-enabled GPU. Running it purely on CPU will be extremely slow.
*   **Length Constraints:** If a product has too many reviews, they will be split into multiple batches, potentially resulting in slightly fragmented summaries across different batches.

---

## 📜 License
This project is licensed under the MIT License - free to use, modify, and distribute.