
# Improved cleaner that preserves sentiment cues, flags short or templated reviews, and avoids over-aggressive language filtering.
import pandas as pd
import re, unicodedata, os
from tqdm import tqdm

INPUT_FILE = "data/raw_reviews_improved.csv"
OUTPUT_FILE = "data/cleaned_reviews_improved.csv"
AUDIT_FILE = "data/cleaning_audit.csv"

MIN_WORDS = 3
MIN_CHARS = 8
SHORT_GENERIC = {
    "bagus", "jelek", "mantap", "buruk", "lumayan", "keren", "parah", "oke", "ok", "good", "bad"
}

tqdm.pandas()


def normalize_text(text_val):
    if pd.isna(text_val):
        return ""
    text_val = str(text_val)
    text_val = unicodedata.normalize("NFKC", text_val)
    text_val = text_val.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text_val = re.sub(r"http\S+|www\.\S+", " ", text_val)
    text_val = re.sub(r"([!?.,])\1{2,}", r"\1\1", text_val)
    text_val = re.sub(r"(.)\1{4,}", r"\1\1", text_val)
    text_val = re.sub(r"\s+", " ", text_val).strip().lower()
    return text_val


def is_short_generic(text_val):
    split_words = text_val.split()
    if len(split_words) <= 2 and text_val in SHORT_GENERIC:
        return True
    return False


def has_enough_content(text_val):
    split_words = text_val.split()
    if len(text_val) < MIN_CHARS:
        return False
    if len(split_words) < MIN_WORDS:
        return False
    return True


def repeated_token_ratio(text_val):
    split_words = text_val.split()
    if len(split_words) == 0:
        return 1.0
    unique_ratio = len(set(split_words)) / len(split_words)
    return 1.0 - unique_ratio


def run_cleaning():
    df_reviews = pd.read_csv(INPUT_FILE)
    print(df_reviews.head())
    print(len(df_reviews))

    df_reviews = df_reviews.drop_duplicates(subset=["review_id"]).copy()
    df_reviews["text"] = df_reviews["text_raw"].progress_apply(normalize_text)
    df_reviews["char_len"] = df_reviews["text"].str.len()
    df_reviews["word_len"] = df_reviews["text"].str.split().str.len()
    df_reviews["short_generic_flag"] = df_reviews["text"].progress_apply(is_short_generic)
    df_reviews["repeat_ratio"] = df_reviews["text"].progress_apply(repeated_token_ratio)
    df_reviews["content_ok"] = df_reviews["text"].progress_apply(has_enough_content)

    df_reviews["text_dedup_key"] = df_reviews["app_id"].astype(str) + " || " + df_reviews["text"]
    df_reviews = df_reviews.drop_duplicates(subset=["text_dedup_key"]).copy()

    df_reviews["keep"] = True
    df_reviews.loc[df_reviews["content_ok"] == False, "keep"] = False
    df_reviews.loc[df_reviews["short_generic_flag"] == True, "keep"] = False
    df_reviews.loc[df_reviews["repeat_ratio"] > 0.65, "keep"] = False

    audit_cols = ["app_name", "rating", "text_raw", "text", "char_len", "word_len", "short_generic_flag", "repeat_ratio", "content_ok", "keep"]
    df_reviews[audit_cols].to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    df_clean = df_reviews[df_reviews["keep"] == True].copy()
    keep_cols = ["app_id", "app_name", "review_id", "text_raw", "text", "rating", "thumbs_up", "review_created", "reply_content", "replied_at", "sort_source"]
    df_clean = df_clean[keep_cols].reset_index(drop=True)
    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(df_clean.head())
    print(df_clean["rating"].value_counts().sort_index())
    print(OUTPUT_FILE)
    print(AUDIT_FILE)
    return df_clean


if __name__ == "__main__":
    df_clean = run_cleaning()
