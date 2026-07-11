"""
Loads a real category from the Amazon Reviews 2023 dataset (McAuley Lab,
UCSD, hosted on Hugging Face) and converts it into the ratings.csv /
products.csv format this project expects.

Categories available (pick a SMALL one to start -- some are huge):
All_Beauty, Appliances, Arts_Crafts_and_Sewing, Automotive, Baby_Products,
Beauty_and_Personal_Care, Books, CDs_and_Vinyl, Cell_Phones_and_Accessories,
Clothing_Shoes_and_Jewelry, Digital_Music, Electronics, Gift_Cards,
Grocery_and_Gourmet_Food, Handmade_Products, Health_and_Household,
Health_and_Personal_Care, Home_and_Kitchen, Industrial_and_Scientific,
Kindle_Store, Magazine_Subscriptions, Movies_and_TV, Musical_Instruments,
Office_Products, Patio_Lawn_and_Garden, Pet_Supplies, Software,
Sports_and_Outdoors, Subscription_Boxes, Tools_and_Home_Improvement,
Toys_and_Games, Video_Games

"All_Beauty" is one of the smallest categories -- good for a first try.

Usage:
    pip install datasets --quiet
    python src/load_real_amazon_data.py --category All_Beauty --max_rows 20000
"""

import argparse
import pandas as pd


def load_category(category: str, max_rows: int = 20000):
    from datasets import load_dataset

    print(f"Downloading '{category}' reviews (this can take a minute)...")
    reviews = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{category}",
        split="full",
        trust_remote_code=True,
        streaming=True,  # avoids downloading the entire split into memory at once
    )

    rows = []
    for i, row in enumerate(reviews):
        if i >= max_rows:
            break
        rows.append({
            "user_id": row["user_id"],
            "product_id": row["parent_asin"],
            "rating": row["rating"],
            "timestamp": row["timestamp"],
        })
    ratings_df = pd.DataFrame(rows)

    print(f"Downloading '{category}' product metadata...")
    meta = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_meta_{category}",
        split="full",
        trust_remote_code=True,
        streaming=True,
    )

    # Only keep metadata for products that actually appear in our ratings sample
    needed_ids = set(ratings_df["product_id"].unique())
    meta_rows = []
    for row in meta:
        if row["parent_asin"] in needed_ids:
            meta_rows.append({
                "product_id": row["parent_asin"],
                "product_name": (row["title"] or "Unknown")[:120],  # trim very long titles
                "category": category,
            })
    products_df = pd.DataFrame(meta_rows).drop_duplicates(subset="product_id")

    return ratings_df, products_df


def clean_and_save(ratings_df, products_df):
    # Drop duplicate user-product ratings, keep the first
    ratings_df = ratings_df.drop_duplicates(subset=["user_id", "product_id"], keep="first")

    # Only keep ratings for products we have metadata for
    ratings_df = ratings_df[ratings_df["product_id"].isin(products_df["product_id"])]

    # Convert timestamp (milliseconds since epoch, as provided by this dataset) to datetime
    ratings_df["timestamp"] = pd.to_datetime(ratings_df["timestamp"], unit="ms")

    ratings_df.to_csv("data/ratings.csv", index=False)
    products_df.to_csv("data/products.csv", index=False)

    print(f"\nSaved {len(ratings_df)} ratings and {len(products_df)} products.")
    print(f"Users: {ratings_df['user_id'].nunique()}, Products: {ratings_df['product_id'].nunique()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="All_Beauty", help="Amazon category name, e.g. All_Beauty")
    parser.add_argument("--max_rows", type=int, default=20000, help="Max number of reviews to pull (keep small at first)")
    args = parser.parse_args()

    ratings_df, products_df = load_category(args.category, args.max_rows)
    clean_and_save(ratings_df, products_df)
