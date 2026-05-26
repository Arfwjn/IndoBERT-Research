from datasets import Dataset, DatasetDict, ClassLabel, Features, Value
import pandas as pd


def export_to_hf_format():
    train = pd.read_csv("../Datasets/data/train_improved.csv")
    val   = pd.read_csv("../Datasets/data/val_improved.csv")
    test  = pd.read_csv("../Datasets/data/test_improved.csv")

    # Verifikasi label
    print("Verifikasi distribusi label:")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:8s}: {dict(df['label'].value_counts().sort_index())}")

    # Pastikan tidak ada nilai selain 0 dan 1
    all_labels = set(train["label"]) | set(val["label"]) | set(test["label"])
    assert all_labels == {0, 1}, (
        f"Ditemukan nilai label tidak terduga: {all_labels}. "
    )
    print("Label valid: hanya 0 (negatif) dan 1 (positif)\n")

    features = Features({
        "text"     : Value("string"),
        "label"    : ClassLabel(names=["negatif", "positif"]),
        "app_name" : Value("string"),
    })

    def to_hf(df):
        return Dataset.from_dict(
            {
                "text"     : df["text"].tolist(),
                "label"    : df["label"].tolist(),  
                "app_name" : df["app_name"].tolist(),
            },
            features=features,
        )

    dataset_dict = DatasetDict({
        "train"     : to_hf(train),
        "validation": to_hf(val),
        "test"      : to_hf(test),
    })

    # Simpan
    dataset_dict.save_to_disk("../Datasets/data/hf_dataset")
    print("Dataset tersimpan di ../Datasets/data/hf_dataset")
    print(dataset_dict)

    print("\nDistribusi label sesudah export:")
    for split in ["train", "validation", "test"]:
        from collections import Counter
        counts = Counter(dataset_dict[split]["label"])
        names  = dataset_dict[split].features["label"].names
        readable = {names[k]: v for k, v in sorted(counts.items())}
        print(f"  {split:12s}: {readable}")

    print("="*30)
    print("\n# Upload ke HuggingFace Hub (opsional):")
    print("# from huggingface_hub import login")
    print("# login()")
    print("# dataset_dict.push_to_hub('username/IndoSentiment-PlayStore')")

    return dataset_dict


if __name__ == "__main__":
    ds = export_to_hf_format()