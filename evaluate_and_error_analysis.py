import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data

def run_evaluation(model_dir="./saved_models/berturk", model_tag="berturk", max_len=128):
    print(f"\n--- {model_tag.upper()} Degerlendirme ve Hata Analizi ---")
    
    if not os.path.exists(model_dir):
        print(f"Hata: {model_dir} dizininde kayitli model bulunamadi. Once train.py calistirin.")
        return

    _, test_df, label2id, id2label = get_data()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    all_preds = []
    all_probs = []

    print("Test verisi tahmin ediliyor...")
    batch_size = 32
    for i in range(0, len(test_df), batch_size):
        batch_texts = test_df["text"].iloc[i:i+batch_size].tolist()
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y_true = test_df["label_id"].tolist()
    y_pred = all_preds
    labels = [id2label[i] for i in range(len(id2label))]

    # 1. Siniflandirma Raporu
    print("\nSINIFLANDIRMA RAPORU:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    # 2. Confusion Matrix cizimi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - {model_tag.upper()}")
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Gercek")
    plt.tight_layout()
    cm_path = f"confusion_matrix_{model_tag}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix kaydedildi: {cm_path}")

    # 3. Yanlis Siniflandirilan Ornekler (Hata Analizi)
    test_df["pred_id"] = y_pred
    test_df["pred_label"] = test_df["pred_id"].map(id2label)
    test_df["confidence"] = [probs[p] for probs, p in zip(all_probs, y_pred)]

    errors_df = test_df[test_df["label_id"] != test_df["pred_id"]].copy()
    error_csv_path = f"misclassified_examples_{model_tag}.csv"
    errors_df[["text", "label", "pred_label", "confidence"]].to_csv(error_csv_path, index=False, encoding="utf-8-sig")
    print(f"Yanlis siniflandirilan {len(errors_df)} ornek kaydedildi: {error_csv_path}")

    print("\nOrnek 3 Hatali Tahmin:")
    for _, row in errors_df.head(3).iterrows():
        print(f"- Metin: {row['text'][:90]}...")
        print(f"  Gercek: {row['label']} | Tahmin: {row['pred_label']} | Guven: {row['confidence']:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_tag", type=str, default="berturk", choices=["berturk", "mbert"])
    args = parser.parse_args()

    run_evaluation(
        model_dir=f"./saved_models/{args.model_tag}",
        model_tag=args.model_tag
    )
