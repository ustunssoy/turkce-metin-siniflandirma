import os
import argparse
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
from data_loader import get_data

MODEL_MAP = {
    "berturk": "dbmdz/bert-base-turkish-cased",
    "mbert": "bert-base-multilingual-cased"
}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

def train_model(model_key="berturk", epochs=3, batch_size=16, lr=2e-5, max_len=128):
    model_name = MODEL_MAP[model_key]
    output_dir = f"./saved_models/{model_key}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- {model_key.upper()} Egitimi Baslatiliyor ({model_name}) ---")
    train_df, test_df, label2id, id2label = get_data()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_len)

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    test_ds = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "label"}))

    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    test_tokenized = test_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=f"./results/{model_key}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=1,
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"\n{model_key.upper()} Test Sonuclari:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Modeli ve tokenizer'i kaydet
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model kaydedildi: {output_dir}")

    return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="berturk", choices=["berturk", "mbert"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_model(
        model_key=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
