from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
import pandas as pd

def export_to_hf_format():
    train = pd.read_csv("../Datasets/data/train_improved.csv")
    val   = pd.read_csv("../Datasets/data/val_improved.csv")
    test  = pd.read_csv("../Datasets/data/test_improved.csv")

    # Definisi schema dataset
    features = Features({
        "text"      : Value("string"),
        "label"     : ClassLabel(names=["negatif", "netral", "positif"]),
        "app_name"  : Value("string"),
    })

    def to_hf(df):
        return Dataset.from_dict(
            {
                "text"    : df["text"].tolist(),
                "label"   : df["label"].tolist(),
                "app_name": df["app_name"].tolist(),
            },
            features=features,
        )

    dataset_dict = DatasetDict({
        "train"     : to_hf(train),
        "validation": to_hf(val),
        "test"      : to_hf(test),
    })

    # Simpan lokal dalam format Arrow
    dataset_dict.save_to_disk("../Datasets/data/hf_dataset")
    print("Dataset tersimpan di ../Datasets/data/hf_dataset")
    print(dataset_dict)

    print("=*"*50)

    # Cara load ulang di notebook lain
    print("\n# Cara load ulang:")
    print("from datasets import load_from_disk")
    print("dataset = load_from_disk('../Datasets/data/hf_dataset')")

    print("=*"*50)

    # Upload ke HuggingFace Hub (Opsional)
    print("\n# Jika ingin upload ke HuggingFace Hub (opsional, gratis):")
    print("# from huggingface_hub import login")
    print("# login()")
    print("# dataset_dict.push_to_hub('username/IndoSentiment-PlayStore')")

    print("=*"*50)

    return dataset_dict


if __name__ == "__main__":
    ds = export_to_hf_format()