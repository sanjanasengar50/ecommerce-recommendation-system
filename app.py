"""
Streamlit demo: enter a user_id and see top product recommendations
from both the popularity baseline and the collaborative filtering model.

Run with: streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from data_loader import load_ratings, load_products, basic_stats
from popularity_recommender import get_popular_products
from collaborative_filtering import train_model, recommend_for_user

st.set_page_config(page_title="E-commerce Recommender", layout="centered")
st.title("🛒 E-commerce Product Recommender")
st.caption("Demo built with popularity baseline + collaborative filtering (SVD)")

@st.cache_data
def get_data():
    return load_ratings(), load_products()

@st.cache_resource
def get_model(ratings):
    return train_model(ratings)

ratings, products = get_data()
stats = basic_stats(ratings)

with st.expander("Dataset stats"):
    st.json(stats)

model, rmse = get_model(ratings)
st.caption(f"Collaborative filtering model RMSE: {rmse:.3f}")

tab1, tab2 = st.tabs(["Personalized (CF)", "Popular products"])

with tab1:
    user_ids = sorted(ratings["user_id"].unique())
    selected_user = st.selectbox("Choose a user", user_ids)
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Get recommendations"):
        recs = recommend_for_user(model, ratings, products, selected_user, top_n=top_n)
        st.table(recs)

with tab2:
    top_n_pop = st.slider("Number of popular products", 3, 10, 5, key="pop_slider")
    pop = get_popular_products(ratings, products, top_n=top_n_pop)
    st.table(pop)
