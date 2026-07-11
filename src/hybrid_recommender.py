"""
Level 4: Hybrid Recommender.

Combines Collaborative Filtering (CF) and Content-Based Filtering (CBF)
to get the strengths of both:
- CF: personalized, learns from user behavior patterns
- CBF: works even for new users/products (solves cold-start)

Strategy used here:
1. If the user has enough rating history -> blend CF + CBF scores.
2. If the user is new / has very few ratings -> fall back to CBF
   seeded from their one or two ratings, or popularity if they have none.
"""

import numpy as np

from data_loader import load_ratings, load_products
from collaborative_filtering import train_model, recommend_for_user as cf_recommend
from content_based_recommender import build_similarity_matrix, get_similar_products
from popularity_recommender import get_popular_products

MIN_RATINGS_FOR_CF = 5  # below this, we don't trust CF for this user


def normalize(scores_dict):
    """Min-max normalize a dict of {id: score} to the 0-1 range."""
    if not scores_dict:
        return {}
    values = list(scores_dict.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: 1.0 for k in scores_dict}
    return {k: (v - lo) / (hi - lo) for k, v in scores_dict.items()}


def hybrid_recommend(user_id, ratings_df, products_df, cf_model, similarity_matrix,
                      cbf_products_df, top_n=5, cf_weight=0.6, cbf_weight=0.4):

    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    n_user_ratings = len(user_ratings)

    # Case 1: brand-new user, no ratings at all -> fall back to popularity
    if n_user_ratings == 0:
        pop = get_popular_products(ratings_df, products_df, top_n=top_n)
        pop["source"] = "popularity (cold-start: no ratings yet)"
        return pop.to_dict("records")

    all_products = products_df["product_id"].tolist()
    already_rated = set(user_ratings["product_id"])
    candidates = [p for p in all_products if p not in already_rated]

    # ---- CF scores (only trustworthy with enough history) ----
    cf_scores = {}
    if n_user_ratings >= MIN_RATINGS_FOR_CF:
        for p in candidates:
            cf_scores[p] = cf_model.predict(user_id, p).est
    cf_scores_norm = normalize(cf_scores)

    # ---- CBF scores: average similarity to products the user already liked (rating >= 4) ----
    liked_products = user_ratings[user_ratings["rating"] >= 4]["product_id"].tolist()
    cbf_scores = {p: 0.0 for p in candidates}
    if liked_products:
        for liked in liked_products:
            try:
                similar = get_similar_products(liked, cbf_products_df, similarity_matrix, top_n=len(candidates))
                for s in similar:
                    if s["product_id"] in cbf_scores:
                        cbf_scores[s["product_id"]] += s["similarity_score"]
            except ValueError:
                continue
        # average over number of liked products
        cbf_scores = {k: v / len(liked_products) for k, v in cbf_scores.items()}
    cbf_scores_norm = normalize(cbf_scores)

    # ---- Combine ----
    final_scores = {}
    for p in candidates:
        cf_s = cf_scores_norm.get(p, 0.0)
        cbf_s = cbf_scores_norm.get(p, 0.0)
        if n_user_ratings >= MIN_RATINGS_FOR_CF:
            final_scores[p] = cf_weight * cf_s + cbf_weight * cbf_s
        else:
            # not enough history for CF -> rely fully on content-based
            final_scores[p] = cbf_s

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for product_id, score in ranked:
        name = products_df.loc[products_df["product_id"] == product_id, "product_name"].values[0]
        results.append({
            "product_id": product_id,
            "product_name": name,
            "hybrid_score": round(score, 3),
            "source": "CF+CBF blend" if n_user_ratings >= MIN_RATINGS_FOR_CF else "CBF only (few ratings)",
        })
    return results


if __name__ == "__main__":
    ratings = load_ratings()
    products = load_products()

    cf_model, rmse = train_model(ratings)
    print(f"CF model RMSE: {rmse:.3f}")

    similarity_matrix, cbf_products_df = build_similarity_matrix(products)

    # Try an existing user with rating history
    sample_user = ratings["user_id"].iloc[0]
    print(f"\nHybrid recommendations for existing user '{sample_user}':")
    for r in hybrid_recommend(sample_user, ratings, products, cf_model, similarity_matrix, cbf_products_df):
        print(r)

    # Try a brand-new user (cold start)
    print(f"\nHybrid recommendations for a brand-new user (cold start):")
    for r in hybrid_recommend("brand_new_user_999", ratings, products, cf_model, similarity_matrix, cbf_products_df):
        print(r)
