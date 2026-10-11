from transformers import AutoTokenizer

def compare_tokenizers():
    samples = [
        "Cumhurbaşkanlığı İletişim Başkanlığı yeni bir açıklama yaptı.",
        "Merkez Bankası faiz kararını açıkladı ve piyasalar hareketlendi.",
        "Galatasaray derbide Fenerbahçe karşısında önemli bir galibiyet aldı.",
        "Yapay zeka modellerinin Türkçe morfolojik yapısını öğrenmesi zorlayıcı olabiliyor."
    ]

    mbert_name = "bert-base-multilingual-cased"
    berturk_name = "dbmdz/bert-base-turkish-cased"

    print("Tokenizer'lar yukleniyor...")
    tok_mbert = AutoTokenizer.from_pretrained(mbert_name)
    tok_berturk = AutoTokenizer.from_pretrained(berturk_name)

    print("\n" + "=" * 60)
    print("TOKENIZER KARSILASTIRMASI (mBERT vs BERTurk)")
    print("=" * 60)

    for i, text in enumerate(samples, 1):
        tokens_mbert = tok_mbert.tokenize(text)
        tokens_berturk = tok_berturk.tokenize(text)

        print(f"\n[Metin {i}]: {text}")
        print(f"mBERT Token Sayisi   : {len(tokens_mbert)}")
        print(f"mBERT Token'lari     : {tokens_mbert}")
        print(f"BERTurk Token Sayisi : {len(tokens_berturk)}")
        print(f"BERTurk Token'lari   : {tokens_berturk}")

    print("\n" + "=" * 60)
    print("Ozet: BERTurk Turkce ek ve kokleri daha butun parcalar halinde saklarken,")
    print("mBERT Turkce metinleri daha kucuk alt birimlere (subwords) bolmektedir.")
    print("=" * 60)

if __name__ == "__main__":
    compare_tokenizers()
