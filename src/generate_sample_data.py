"""
Generates a small synthetic dataset that mimics the structure of the
Amazon Product Reviews dataset (user_id, product_id, rating, timestamp).

Use this to test your pipeline end-to-end before plugging in the real
Amazon dataset (see README.md for the real dataset link).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_USERS = 200
N_PRODUCTS = 50
N_RATINGS = 3000

PRODUCT_NAMES = [f"Product_{i}" for i in range(N_PRODUCTS)]
PRODUCT_CATEGORIES = np.random.choice(
    ["Electronics", "Books", "Home", "Sports", "Beauty"], size=N_PRODUCTS
)

def generate():
    user_ids = np.random.choice([f"user_{i}" for i in range(N_USERS)], N_RATINGS)
    product_idx = np.random.choice(range(N_PRODUCTS), N_RATINGS)
    product_ids = [f"product_{i}" for i in product_idx]

    # Ratings skewed toward positive, like real e-commerce data
    ratings = np.random.choice([1, 2, 3, 4, 5], N_RATINGS, p=[0.05, 0.05, 0.15, 0.35, 0.4])

    timestamps = pd.date_range("2023-01-01", periods=N_RATINGS, freq="h")

    df = pd.DataFrame({
        "user_id": user_ids,
        "product_id": product_ids,
        "rating": ratings,
        "timestamp": timestamps,
    })

    # Drop duplicate user-product pairs, keep first
    df = df.drop_duplicates(subset=["user_id", "product_id"], keep="first")

    products_df = pd.DataFrame({
        "product_id": [f"product_{i}" for i in range(N_PRODUCTS)],
        "product_name": PRODUCT_NAMES,
        "category": PRODUCT_CATEGORIES,
    })

    df.to_csv("data/ratings.csv", index=False)
    products_df.to_csv("data/products.csv", index=False)
    print(f"Generated {len(df)} ratings for {N_USERS} users and {N_PRODUCTS} products.")
    print("Saved to data/ratings.csv and data/products.csv")

if __name__ == "__main__":
    generate()
