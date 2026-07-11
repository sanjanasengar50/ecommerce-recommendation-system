"""
Loads ratings and product metadata from data/ folder.

Expected ratings.csv columns: user_id, product_id, rating, timestamp
Expected products.csv columns: product_id, product_name, category
"""

import pandas as pd


def load_ratings(path="data/ratings.csv"):
    df = pd.read_csv(path)
    required_cols = {"user_id", "product_id", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"ratings.csv is missing required columns: {missing}")
    return df


def load_products(path="data/products.csv"):
    return pd.read_csv(path)


def basic_stats(ratings_df):
    return {
        "n_users": ratings_df["user_id"].nunique(),
        "n_products": ratings_df["product_id"].nunique(),
        "n_ratings": len(ratings_df),
        "avg_rating": round(ratings_df["rating"].mean(), 2),
        "sparsity_pct": round(
            100 * (1 - len(ratings_df) / (
                ratings_df["user_id"].nunique() * ratings_df["product_id"].nunique()
            )), 2
        ),
    }


if __name__ == "__main__":
    ratings = load_ratings()
    print(basic_stats(ratings))
