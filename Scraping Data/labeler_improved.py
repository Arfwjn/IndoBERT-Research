
# Improved labeler with ambiguity control, app-aware deduplication, and split before balancing to reduce leakage.
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "../Datasets/data/cleaned_reviews_improved.csv"
OUTPUT_TRAIN = "../Datasets/data/train_improved.csv"
OUTPUT_VAL = "../Datasets/data/val_improved.csv"
OUTPUT_TEST = "../Datasets/data/test_improved.csv"
OUTPUT_FULL = "../Datasets/data/dataset_full_improved.csv"
OUTPUT_AUDIT = "../Datasets/data/label_audit_improved.csv"

LABEL_MAP = {1: 0, 2: 0, 4: 2, 5: 2}
LABEL_NAMES = {0: "negatif", 2: "positif"}


def mark_ambiguous(text_val, rating_val):
    text_val = str(text_val)
    mixed_terms = ["tapi", "tp", "namun", "cuma", "meski", "walau", "walaupun"]
    has_mixed = any(term in text_val.split() for term in mixed_terms)
    if rating_val == 3:
        return True
    if has_mixed and len(text_val.split()) >= 5:
        return True
    return False


def assign_label(row_val):
    rating_val = int(row_val["rating"])
    if rating_val in LABEL_MAP:
        return LABEL_MAP[rating_val]
    return None


def stratified_split(df_input):
    train_df, temp_df = train_test_split(
        df_input,
        test_size=0.30,
        random_state=42,
        stratify=df_input["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def balance_only_train(train_df):
    counts_val = train_df["label"].value_counts()
    min_count_val = counts_val.min()
    sampled_dfs = [group_df.sample(min_count_val, random_state=42) for label_val, group_df in train_df.groupby("label")]
    balanced_train = pd.concat(sampled_dfs).reset_index(drop=True)
    return balanced_train.sample(frac=1, random_state=42).reset_index(drop=True)


def run_labeling():
    df_clean = pd.read_csv(INPUT_FILE)
    df_clean["ambiguous_flag"] = df_clean.apply(lambda row_val: mark_ambiguous(row_val["text"], row_val["rating"]), axis=1)
    df_clean["label"] = df_clean.apply(assign_label, axis=1)
    df_clean["label_name"] = df_clean["label"].map(LABEL_NAMES)

    audit_cols = ["app_name", "rating", "text_raw", "text", "ambiguous_flag", "label", "label_name"]
    df_clean[audit_cols].to_csv(OUTPUT_AUDIT, index=False, encoding="utf-8-sig")

    df_model = df_clean[(df_clean["ambiguous_flag"] == False) & (df_clean["label"].notna())].copy()
    df_model = df_model.drop_duplicates(subset=["app_id", "text"]).reset_index(drop=True)

    train_df, val_df, test_df = stratified_split(df_model)
    train_balanced = balance_only_train(train_df)

    save_cols = ["text", "label", "label_name", "app_name", "app_id", "rating", "thumbs_up", "review_id"]
    train_balanced[save_cols].to_csv(OUTPUT_TRAIN, index=False, encoding="utf-8-sig")
    val_df[save_cols].to_csv(OUTPUT_VAL, index=False, encoding="utf-8-sig")
    test_df[save_cols].to_csv(OUTPUT_TEST, index=False, encoding="utf-8-sig")
    df_model[save_cols].to_csv(OUTPUT_FULL, index=False, encoding="utf-8-sig")

    print(train_balanced.head())
    print(train_balanced["label_name"].value_counts())
    print(val_df["label_name"].value_counts())
    print(test_df["label_name"].value_counts())
    print(OUTPUT_TRAIN)
    print(OUTPUT_VAL)
    print(OUTPUT_TEST)
    print(OUTPUT_FULL)
    print(OUTPUT_AUDIT)
    return train_balanced, val_df, test_df, df_model


if __name__ == "__main__":
    train_balanced, val_df, test_df, df_model = run_labeling()
