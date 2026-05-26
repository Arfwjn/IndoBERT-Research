# Improved scraper with pagination, mixed sort strategy, metadata retention, and raw text preservation.
from google_play_scraper import reviews, Sort
from tqdm import tqdm
import pandas as pd
import time, os, re

TARGET_APPS = {
    "com.tokopedia.tkpd": "Tokopedia",
    "com.shopee.id": "Shopee",
    "com.bukalapak.android": "Bukalapak",
    "com.lazada.android": "Lazada ID",
    "com.gojek.app": "Gojek",
    "com.grabtaxi.passenger": "Grab",
    "id.dana": "DANA",
    "com.bca": "BCA Mobile",
    "org.detikcom.rss": "detikcom",
    "com.linkdokter.halodoc.android": "Halodoc",
}

REVIEWS_PER_APP = 2000
OUTPUT_RAW = "../Datasets/data/raw_reviews_improved.csv"
os.makedirs("../Datasets/data", exist_ok=True)

SORT_PLAN = [
    (Sort.NEWEST, 0.55),
    (Sort.MOST_RELEVANT, 0.45),
]


def safe_text(value):
    if value is None:
        return ""
    return str(value).replace("\x00", " ").strip()


def fetch_reviews_chunk(app_id, app_name, score_value, sort_value, target_count):
    rows = []
    token_val = None
    seen_ids = set()

    while len(rows) < target_count:
        batch_size = min(200, target_count - len(rows))
        result, token_val = reviews(
            app_id,
            lang="id",
            country="id",
            sort=sort_value,
            count=batch_size,
            continuation_token=token_val,
            filter_score_with=score_value,
        )
        if not result:
            break
        for item in result:
            review_id_val = item.get("reviewId", "")
            if review_id_val in seen_ids:
                continue
            seen_ids.add(review_id_val)
            rows.append({
                "app_id": app_id,
                "app_name": app_name,
                "review_id": review_id_val,
                "text_raw": safe_text(item.get("content", "")),
                "rating": int(item.get("score", 0)),
                "thumbs_up": int(item.get("thumbsUpCount", 0)),
                "review_created": str(item.get("at", "")),
                "reply_content": safe_text(item.get("replyContent", "")),
                "replied_at": str(item.get("repliedAt", "")),
                "sort_source": str(sort_value),
                "score_filter": score_value,
            })
        if token_val is None:
            break
        time.sleep(0.2)
    return rows


def scrape_app(app_id, app_name, total_target):
    rows = []
    seen_ids = set()
    per_star_total = max(total_target // 5, 1)

    for score_value in range(1, 6):
        for sort_value, weight_val in SORT_PLAN:
            target_piece = max(int(per_star_total * weight_val), 1)
            try:
                piece_rows = fetch_reviews_chunk(app_id, app_name, score_value, sort_value, target_piece)
                for row_val in piece_rows:
                    if row_val["review_id"] in seen_ids:
                        continue
                    seen_ids.add(row_val["review_id"])
                    rows.append(row_val)
            except Exception as error_val:
                print("Error app " + app_name + " score " + str(score_value) + " sort " + str(sort_value) + " -> " + str(error_val))
            time.sleep(0.4)
    return rows


def run_scraping():
    all_rows = []
    for app_id, app_name in tqdm(TARGET_APPS.items()):
        app_rows = scrape_app(app_id, app_name, REVIEWS_PER_APP)
        all_rows.extend(app_rows)
        print(app_name + " collected " + str(len(app_rows)))
        time.sleep(0.8)

    df_reviews = pd.DataFrame(all_rows)
    if len(df_reviews) == 0:
        raise ValueError("No reviews collected")

    df_reviews = df_reviews.drop_duplicates(subset=["review_id"]).reset_index(drop=True)
    df_reviews.to_csv(OUTPUT_RAW, index=False, encoding="utf-8-sig")
    print(df_reviews.head())
    print(df_reviews["rating"].value_counts().sort_index())
    print(OUTPUT_RAW)
    return df_reviews


if __name__ == "__main__":
    df_reviews = run_scraping()
