"""
Level 5: Proper evaluation for a recommender system.

Two upgrades over Level 2's basic train_test_split + RMSE:

1. TIME-BASED SPLIT instead of random split.
   Random split lets the model "see the future" during training (e.g. a
   rating from January could end up in train while a rating from last
   week ends up in test). Real deployment never works that way: you only
   ever have past data to predict future behavior. So we sort by
   timestamp and put the earliest ~80% in train, latest ~20% in test.

2. PRECISION@K / RECALL@K instead of just RMSE.
   RMSE measures "how close was the predicted rating number" -- but
   users don't see predicted numbers, they see a top-K list. What
   actually matters: of the top-K products we recommended, how many
   did the user actually like? That's Precision@K. Of all the products
   the user actually liked, how many did we manage to surface in our
   top-K? That's Recall@K.
"""

from collections import defaultdict

from data_loader import load_ratings
from collaborative_filtering import SVDModel

RELEVANCE_THRESHOLD = 4  # a rating >= 4 counts as "the user liked it"


def time_based_split(ratings_df, test_fraction=0.2):
    """
    Sorts all ratings by timestamp, then takes the earliest portion as
    train and the most recent portion as test. This simulates a real
    deployment: predict future behavior using only past behavior.
    """
    df = ratings_df.sort_values("timestamp").reset_index(drop=True)
    split_point = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_point]
    test_df = df.iloc[split_point:]
    return train_df, test_df


def train_svd_on_split(train_df, n_factors=20):
    return SVDModel(n_factors=n_factors).fit(train_df)


def precision_recall_at_k(model, test_df, k=5, threshold=RELEVANCE_THRESHOLD):
    """
    For every user in the test set, predicts a score for every product
    they actually rated in that period, ranks those predictions, takes
    the top-K, and checks how many of those top-K were truly "liked"
    (actual rating >= threshold).
    """
    user_predictions = defaultdict(list)

    for row in test_df.itertuples():
        pred = model.predict(row.user_id, row.product_id)
        user_predictions[row.user_id].append((pred.est, row.rating))

    precisions, recalls = [], []

    for user_id, preds in user_predictions.items():
        # Rank this user's test items by predicted score, highest first
        preds.sort(key=lambda x: x[0], reverse=True)
        top_k = preds[:k]

        n_relevant_total = sum(1 for (_, actual) in preds if actual >= threshold)
        n_relevant_in_top_k = sum(1 for (_, actual) in top_k if actual >= threshold)

        precisions.append(n_relevant_in_top_k / len(top_k) if top_k else 0)
        recalls.append(n_relevant_in_top_k / n_relevant_total if n_relevant_total > 0 else 0)

    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    return avg_precision, avg_recall


if __name__ == "__main__":
    ratings = load_ratings()

    train_df, test_df = time_based_split(ratings, test_fraction=0.2)
    print(f"Train: {len(train_df)} ratings (up to {train_df['timestamp'].max()})")
    print(f"Test:  {len(test_df)} ratings (from {test_df['timestamp'].min()} onward)")

    model = train_svd_on_split(train_df)

    for k in [3, 5, 10]:
        precision, recall = precision_recall_at_k(model, test_df, k=k)
        print(f"\nK={k}")
        print(f"  Precision@{k}: {precision:.3f}  (of the top-{k} we showed, this fraction were actually liked)")
        print(f"  Recall@{k}:    {recall:.3f}  (of everything the user liked, this fraction was captured in top-{k})")
