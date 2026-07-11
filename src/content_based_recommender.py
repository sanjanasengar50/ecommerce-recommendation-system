"""
Level 3: Content-Based Filtering.

Recommends products similar to a given product based on its features
(name + category), not based on user rating history. This solves the
cold-start problem: works even for a brand-new product with zero ratings.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_loader import load_products


def build_similarity_matrix(products_df):
    """
    Converts each product's text (name + category) into a TF-IDF vector,
    then computes pairwise cosine similarity between all products.
    """
    # Combine features into one text field per product.
    # In a real dataset, you'd also add product description, brand, tags, etc.
    products_df = products_df.copy()
    products_df["combined_features"] = (
        products_df["product_name"] + " " + products_df["category"]
    )

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(products_df["combined_features"])

    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix, products_df


def get_similar_products(product_id, products_df, similarity_matrix, top_n=5):
    """
    Given a product_id, returns the top_n most similar products
    based on content (name + category), regardless of any ratings.
    """
    if product_id not in products_df["product_id"].values:
        raise ValueError(f"product_id '{product_id}' not found in products.csv")

    idx = products_df.index[products_df["product_id"] == product_id][0]
    scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity score, skip index 0 (the product itself, always similarity=1.0)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    results = []
    for i, score in scores:
        row = products_df.iloc[i]
        results.append({
            "product_id": row["product_id"],
            "product_name": row["product_name"],
            "category": row["category"],
            "similarity_score": round(score, 3),
        })
    return results


if __name__ == "__main__":
    products = load_products()
    similarity_matrix, products_df = build_similarity_matrix(products)

    sample_product = products_df["product_id"].iloc[0]
    print(f"Products similar to '{sample_product}' ({products_df.iloc[0]['category']}):\n")
    similar = get_similar_products(sample_product, products_df, similarity_matrix)
    for s in similar:
        print(s)
