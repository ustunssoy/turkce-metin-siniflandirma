import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_single(text, model_dir="./saved_models/berturk"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()

    pred_label = model.config.id2label[pred_idx]
    print(f"\nMetin      : {text}")
    print(f"Tahmin     : {pred_label}")
    print(f"Guven Skoru: %{confidence * 100:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Merkez Bankasi faiz politikasinda yeni adimlar acikladi.")
    parser.add_argument("--model_dir", type=str, default="./saved_models/berturk")
    args = parser.parse_args()

    predict_single(args.text, args.model_dir)
