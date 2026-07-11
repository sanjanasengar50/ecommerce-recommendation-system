"""
Level 2: Collaborative Filtering using matrix factorization (SVD).

This learns latent factors for users and products from the ratings
matrix, then predicts ratings for products a user hasn't rated yet.

Implementation note: this uses scikit-learn's TruncatedSVD rather than
the `surprise` library. `surprise` compiles C extensions at install
time and doesn't ship pre-built wheels for every Python version, which
made it unreliable to deploy (e.g. it repeatedly failed to build on
Streamlit Community Cloud). scikit-learn ships pre-built wheels for all
current Python versions, so this approach installs and deploys
reliably everywhere.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error

from data_loader import load_ratings, load_products


class Prediction:
    """Small wrapper so calling code can do `model.predict(u, p).est`,
    matching the interface other modules (e.g. hybrid_recommender.py)
    already expect."""
    def __init__(self, est):
        self.est = est


class SVDModel:
    def __init__(self, n_factors=20):
        self.n_factors = n_factors
        self.user_index = {}
        self.item_index = {}
        self.global_mean = 0.0
        self.user_factors = None
        self.item_factors = None

    def fit(self, ratings_df):
        self.global_mean = ratings_df["rating"].mean()

        users = ratings_df["user_id"].unique()
        items = ratings_df["product_id"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {p: i for i, p in enumerate(items)}

        n_users, n_items = len(users), len(items)

        # Build a user-item matrix, centered around the global mean.
        # Missing (unrated) entries stay at 0, i.e. "average" after centering.
        matrix = np.zeros((n_users, n_items))
        for row in ratings_df.itertuples():
            u = self.user_index[row.user_id]
            i = self.item_index[row.product_id]
            matrix[u, i] = row.rating - self.global_mean

        k = max(1, min(self.n_factors, min(n_users, n_items) - 1))
        svd = TruncatedSVD(n_components=k, random_state=42)
        self.user_factors = svd.fit_transform(matrix)
        self.item_factors = svd.components_.T
        return self

    def predict(self, user_id, product_id):
        if user_id not in self.user_index or product_id not in self.item_index:
            # Unseen user or product -> fall back to the global average rating
            return Prediction(self.global_mean)
        u = self.user_index[user_id]
        i = self.item_index[product_id]
        est = self.global_mean + self.user_factors[u].dot(self.item_factors[i])
        est = min(5, max(1, est))  # keep predictions within the valid rating range
        return Prediction(est)


def train_model(ratings_df, test_size=0.2, n_factors=20):
    """
    Randomly splits ratings into train/test, fits the model on train,
    and reports RMSE on test.
    """
    shuffled = ratings_df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_point = int(len(shuffled) * (1 - test_size))
    train_df = shuffled.iloc[:split_point]
    test_df = shuffled.iloc[split_point:]

    model = SVDModel(n_factors=n_factors).fit(train_df)

    preds = [model.predict(row.user_id, row.product_id).est for row in test_df.itertuples()]
    actuals = test_df["rating"].tolist()
    rmse = mean_squared_error(actuals, preds) ** 0.5

    return model, rmse


def recommend_for_user(model, ratings_df, products_df, user_id, top_n=5):
    all_products = products_df["product_id"].unique()
    already_rated = set(ratings_df[ratings_df["user_id"] == user_id]["product_id"])
    candidates = [p for p in all_products if p not in already_rated]

    preds = [(p, model.predict(user_id, p).est) for p in candidates]
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:top_n]

    result = []
    for product_id, score in top:
        name = products_df.loc[products_df["product_id"] == product_id, "product_name"].values[0]
        result.append({"product_id": product_id, "product_name": name, "predicted_rating": round(score, 2)})
    return result


if __name__ == "__main__":
    ratings = load_ratings()
    products = load_products()

    model, rmse = train_model(ratings)
    print(f"Model trained. RMSE on test set: {rmse:.3f}")

    sample_user = ratings["user_id"].iloc[0]
    recs = recommend_for_user(model, ratings, products, sample_user)
    print(f"\nTop recommendations for {sample_user}:")
    for r in recs:
        print(r)
