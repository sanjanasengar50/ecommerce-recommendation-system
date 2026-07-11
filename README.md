# E-commerce Product Recommendation System

🔗 **Live demo:** https://ecommerce-recommendation-system-apj4an7nnakhnxzpqwl3ti.streamlit.app/

A recommendation engine built for an e-commerce use case (Amazon-style), covering the core approaches used by real recommendation teams: popularity baseline → collaborative filtering → content-based → hybrid → proper evaluation.

## Why this project

Recommendation systems are a core ML problem at companies like Amazon (e.g. "customers who bought this also bought"). This project demonstrates the full pipeline: data loading, baseline modeling, collaborative filtering, content-based filtering, a hybrid model, rigorous evaluation, and a deployed demo — trained and tested on a real subset of Amazon customer review data.

## Project structure

ecommerce-recommender/
├── data/
│   ├── ratings.csv        # user_id, product_id, rating, timestamp
│   └── products.csv       # product_id, product_name, category
├── src/
│   ├── generate_sample_data.py       # creates synthetic test data (for quick local testing)
│   ├── load_real_amazon_data.py      # downloads + formats a real Amazon Reviews 2023 category
│   ├── data_loader.py                # loads + validates CSVs
│   ├── popularity_recommender.py     # Level 1: baseline
│   ├── collaborative_filtering.py    # Level 2: SVD matrix factorization (scikit-learn)
│   ├── content_based_recommender.py  # Level 3: TF-IDF + cosine similarity
│   ├── hybrid_recommender.py         # Level 4: combines CF + content-based
│   └── evaluation.py                 # Level 5: time-based split + Precision@K/Recall@K
├── app.py                 # Streamlit demo
└── requirements.txt

## Setup

```bash
pip install -r requirements.txt
```

## Quickstart (with synthetic sample data)

```bash
cd ecommerce-recommender
python src/generate_sample_data.py     # creates data/ratings.csv and data/products.csv
PYTHONPATH=src python src/popularity_recommender.py
PYTHONPATH=src python src/collaborative_filtering.py
streamlit run app.py
```

## Using the real Amazon dataset

This project can also run on a real subset of the **Amazon Reviews 2023** dataset (McAuley Lab, UCSD) instead of synthetic data:

```bash
python src/load_real_amazon_data.py --category All_Beauty --max_rows 20000
```

This downloads real customer reviews and product metadata directly from McAuley Lab's hosting, and saves them into `data/ratings.csv` and `data/products.csv` in the format the rest of the pipeline expects. Swap `--category` for any category listed at https://amazon-reviews-2023.github.io/ (start with a small one — some categories have tens of millions of reviews).

## Roadmap (what was built, in order)

- [x] Level 1: Popularity-based baseline (bayesian-weighted average rating)
- [x] Level 2: Collaborative filtering (SVD matrix factorization via scikit-learn's `TruncatedSVD` — chosen over the `surprise` library, which relies on C-extension builds that proved unreliable to deploy)
- [x] Level 3: Content-based filtering (TF-IDF + cosine similarity on product name/category)
- [x] Level 4: Hybrid model combining CF + content-based (`src/hybrid_recommender.py`) — switches strategy based on how much rating history a user has, so cold-start users get a sensible fallback instead of an error
- [x] Level 5: Time-based train/test split + Precision@K / Recall@K evaluation (`src/evaluation.py`) — a more realistic simulation of deployment than random split + RMSE alone

## Evaluation notes

- RMSE tells you how close predicted ratings are to actual ratings, but it's **not** the best metric for recommendation quality — a user cares about "are the top-5 recommended products relevant," not the exact predicted rating.
- This project also reports **Precision@K** and **Recall@K** (see Level 5 above) — this is what's actually discussed in industry interviews.

## Known limitation (mention this explicitly in your portfolio)

Pure collaborative filtering suffers from the **cold-start problem**: it can't recommend anything for a brand-new user or a brand-new product with no ratings yet. This is exactly why Level 4 (hybrid) matters — being able to articulate this limitation and how you'd solve it is often more valuable in interviews than the model itself.

apne alfaazon me explain karni aati hai ya nahi — interview se pehle confidence ke liye?
