"""
Level 1: Popularity-based recommender (baseline).

Recommends the products with the highest average rating, weighted by
number of ratings (so a product with 1 five-star rating doesn't beat
a product with 500 ratings averaging 4.5).
"""

from data_loader import load_ratings, load_products


def get_popular_products(ratings_df, products_df, top_n=10, min_ratings=5):
    stats = ratings_df.groupby("product_id")["rating"].agg(["mean", "count"])
    stats = stats[stats["count"] >= min_ratings]

    # Weighted score: bayesian-average style, so products with few ratings
    # don't dominate just by chance
    C = stats["count"].mean()
    m = ratings_df["rating"].mean()
    stats["score"] = (stats["count"] / (stats["count"] + C)) * stats["mean"] + \
                      (C / (stats["count"] + C)) * m

    top = stats.sort_values("score", ascending=False).head(top_n)
    result = top.merge(products_df, left_index=True, right_on="product_id")
    return result[["product_id", "product_name", "category", "mean", "count", "score"]]


if __name__ == "__main__":
    ratings = load_ratings()
    products = load_products()
    print(get_popular_products(ratings, products))
