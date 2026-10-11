import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def get_data(test_size=0.2, random_state=42):
    try:
        # Hugging Face uzerinden TTC veri seti yukleme denemesi
        dataset = load_dataset("savasy/ttc4900", split="train")
        df = pd.DataFrame(dataset)
        # TTC-3600 formati icin text ve category kolonlari
        if "text" in df.columns and "category" in df.columns:
            df = df.rename(columns={"text": "text", "category": "label"})
    except Exception as e:
        print(f"Hugging Face veri seti indirilemedi ({e}), ornek veri seti olusturuluyor...")
        data = {
            "text": [
                "Cumhurbaskani Meclis acilisinda yeni reform paketini duyurdu.",
                "Merkez Bankasi faiz kararini acikladi ve piyasalar hareketlendi.",
                "Galatasaray derbide Fenerbahce karsisinda onemli bir galibiyet aldi.",
                "Yeni yapay zeka cip tasarimi teknolojide yeni bir donem baslatiyor.",
                "Uzmanlar gunde en az iki litre su tuketmenin sagliga faydalarini anlatti.",
                "Antik kent kazilarinda bin bes yuz yillik mozaikler gun yuzune cikarildi."
            ] * 600,
            "label": [
                "siyaset", "ekonomi", "spor", "teknoloji", "saglik", "kultur"
            ] * 600
        }
        df = pd.DataFrame(data)

    # Etiketleri sayisal degerlere donusturme
    unique_labels = sorted(list(df["label"].unique()))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for i, label in enumerate(unique_labels)}
    df["label_id"] = df["label"].map(label2id)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_id"]
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), label2id, id2label

if __name__ == "__main__":
    train_df, test_df, label2id, id2label = get_data()
    print("Egitim boyutu:", len(train_df))
    print("Test boyutu:", len(test_df))
    print("Siniflar:", label2id)
