"""
fine_tune.py
============
Fine-tunes distilbert-base-uncased on the fake vs. real review dataset.

Requirements (install once):
    pip install transformers datasets scikit-learn torch pandas

Run:
    python fine_tune.py
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


# CONFIG
MODEL_NAME   = "distilbert-base-uncased"
DATA_PATH    = "../dataset/reviews.csv"
OUTPUT_DIR   = "../results/model"
LOG_DIR      = "../results/logs"
NUM_LABELS   = 2          # 0 = real, 1 = fake
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 4
SEED         = 42

LABEL2ID     = {"real": 0, "fake": 1}
ID2LABEL     = {0: "real", 1: "fake"}


# 1. LOAD & PREPARE DATA
def load_data(path: str):
    df = pd.read_csv(path)
    df = df[["review_text", "label"]].dropna()
    df["label"] = df["label"].map(LABEL2ID)
    print(f"\n📊 Dataset loaded — {len(df)} total samples")
    print(df["label"].value_counts().rename(index={0: "real", 1: "fake"}).to_string())
    return df


# 2. TOKENIZE
def tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["review_text"],
        truncation=True,
        max_length=MAX_LEN,
    )


# 3. METRICS
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# 4. MAIN
def main():
    print("=" * 55)
    print(" Fine-Tuning Pipeline")
    print("=" * 55)

    # Load data
    df = load_data(DATA_PATH)

    train_df, eval_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=df["label"]
    )
    print(f"\n Train: {len(train_df)} | Eval: {len(eval_df)}")

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    eval_ds  = Dataset.from_pandas(eval_df.reset_index(drop=True))

    # Tokenizer
    print(f"\n Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = train_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)
    eval_ds  = eval_ds.map(lambda b: tokenize_batch(b, tokenizer), batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch",  columns=["input_ids", "attention_mask", "label"])

    # Model
    print(f"\n Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        seed=SEED,
        report_to="none",  
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    #  Train
    print("\n Starting fine-tuning...\n")
    trainer.train()

    #  Evaluate
    print("\n Final evaluation on held-out set:")
    predictions = trainer.predict(eval_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    true_labels = eval_df["label"].values

    print("\n" + "=" * 45)
    print(classification_report(
        true_labels, preds,
        target_names=["real", "fake"]
    ))

    #  Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n Model saved to: {OUTPUT_DIR}")
    print("\n Fine-tuning complete! Run inference.py to test new reviews.")


if __name__ == "__main__":
    main()
