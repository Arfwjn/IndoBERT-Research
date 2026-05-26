import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE    = "../Datasets/data/cleaned_reviews_improved.csv"
OUTPUT_TRAIN  = "../Datasets/data/train_improved.csv"
OUTPUT_VAL    = "../Datasets/data/val_improved.csv"
OUTPUT_TEST   = "../Datasets/data/test_improved.csv"
OUTPUT_FULL   = "../Datasets/data/dataset_full_improved.csv"
OUTPUT_AUDIT  = "../Datasets/data/label_audit_improved.csv"

# Hanya 2 kelas karena rating 3 sudah dibuang di cleaner
# LABEL_MAP = {1: 0, 2: 0, 4: 1, 5: 1}  → menghasilkan nilai 0 dan 1
# Mapping konsisten dengan ClassLabel(names=["negatif", "positif"])
LABEL_MAP   = {1: 0, 2: 0, 4: 1, 5: 1}
LABEL_NAMES = {0: "negatif", 1: "positif"}


def assign_label(row_val):
    rating_val = int(row_val["rating"])
    return LABEL_MAP.get(rating_val, None)


def stratified_split(df_input):
    train_df, temp_df = train_test_split(
        df_input,
        test_size=0.30,
        random_state=42,
        stratify=df_input["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def balance_only_train(train_df):    
    counts_val  = train_df["label"].value_counts()
    min_count   = counts_val.min()
    print(f"\nDistribusi sebelum balancing: {dict(counts_val)}")
    print(f"Min count (dipakai untuk balance): {min_count:,}")

    sampled_dfs = [
        group_df.sample(min_count, random_state=42)
        for _, group_df in train_df.groupby("label")
    ]
    balanced = pd.concat(sampled_dfs).reset_index(drop=True)
    return balanced.sample(frac=1, random_state=42).reset_index(drop=True)


def run_labeling():
    df_clean = pd.read_csv(INPUT_FILE)
    print(f"Total data masuk dari cleaner: {len(df_clean):,}")
    print(f"Distribusi rating:\n{df_clean['rating'].value_counts().sort_index()}\n")

    df_clean["label"]      = df_clean.apply(assign_label, axis=1)
    df_clean["label_name"] = df_clean["label"].map(LABEL_NAMES)

    # Audit sebelum filter
    audit_cols = ["app_name", "rating", "text_raw", "text", "label", "label_name"]
    df_clean[audit_cols].to_csv(OUTPUT_AUDIT, index=False, encoding="utf-8-sig")

    # Hanya simpan baris dengan label valid (0 atau 1)
    df_model = df_clean[df_clean["label"].notna()].copy()
    df_model = df_model.drop_duplicates(subset=["app_id", "text"]).reset_index(drop=True)

    print(f"Total setelah dedup: {len(df_model):,}")
    print(f"Distribusi label:\n{df_model['label_name'].value_counts()}\n")

    # Split stratified
    train_df, val_df, test_df = stratified_split(df_model)

    # Balance hanya pada train
    train_balanced = balance_only_train(train_df)

    # Simpan
    save_cols = [
        "text", "label", "label_name", "app_name",
        "app_id", "rating", "thumbs_up", "review_id",
    ]
    train_balanced[save_cols].to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    val_df[save_cols].to_csv(OUTPUT_VAL,            index=False, encoding="utf-8-sig")
    test_df[save_cols].to_csv(OUTPUT_TEST,           index=False, encoding="utf-8-sig")
    df_model[save_cols].to_csv(OUTPUT_FULL,          index=False, encoding="utf-8-sig")

    print(f"\nHasil split & balancing:")
    print(f"  Train (balanced) : {len(train_balanced):,} | {dict(train_balanced['label_name'].value_counts())}")
    print(f"  Validation       : {len(val_df):,}  | {dict(val_df['label_name'].value_counts())}")
    print(f"  Test             : {len(test_df):,}  | {dict(test_df['label_name'].value_counts())}")
    print(f"\nOutput → {OUTPUT_TRAIN}")
    print(f"         {OUTPUT_VAL}")
    print(f"         {OUTPUT_TEST}")
    print(f"         {OUTPUT_FULL}")
    print(f"         {OUTPUT_AUDIT}")

    return train_balanced, val_df, test_df, df_model


if __name__ == "__main__":
    train_balanced, val_df, test_df, df_model = run_labeling()