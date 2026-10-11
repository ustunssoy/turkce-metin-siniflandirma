# Türkçe Metin Sınıflandırma (BERT vs BERTürk)

Python, PyTorch, Hugging Face, scikit-learn | **GitHub:** ustunssoy/turkce-metin-siniflandirma

- Çok dilli BERT (`bert-base-multilingual-cased`) ve BERTürk (`dbmdz/bert-base-turkish-cased`) modelleri TTC-3600 haber veri seti üzerinde fine-tune edilip karşılaştırıldı.
- **Sonuçlar:** BERTürk ile %94 doğruluk ve 0.94 F1 (çok dilli BERT: %91).
- Tokenizer farklarının morfolojik etkisi incelendi.
- Confusion matrix ve yanlış sınıflandırılan örnekler üzerinden hata analizi yapıldı.

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Kullanım

### 1. Tokenizer Karşılaştırması
```bash
python tokenizer_analysis.py
```

### 2. Model Eğitimi (Fine-Tuning)
BERTürk eğitimi:
```bash
python train.py --model berturk --epochs 3 --batch_size 16
```

Çok Dilli BERT (mBERT) eğitimi:
```bash
python train.py --model mbert --epochs 3 --batch_size 16
```

### 3. Değerlendirme ve Hata Analizi
Confusion matrix görseli ve hatalı örneklerin CSV çıktısı için:
```bash
python evaluate_and_error_analysis.py --model_tag berturk
python evaluate_and_error_analysis.py --model_tag mbert
```

### 4. Tekil Metin Tahmini
```bash
python predict.py --text "Merkez Bankası faiz politikasında yeni adımlar açıkladı."
```
