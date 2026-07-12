# E-commerce Product Recommendation System

🔗 **Live demo:** https://ecommerce-recommendation-system-apj4an7nnakhnxzpqwl3ti.streamlit.app/

A recommendation engine built for an e-commerce use case (Amazon-style), covering the core approaches used by real recommendation teams: popularity baseline → collaborative filtering → content-based → hybrid → proper evaluation.

## Why this project

Recommendation systems are a core ML problem at companies like Amazon (e.g. "customers who bought this also bought"). This project demonstrates the full pipeline: data loading, baseline modeling, collaborative filtering, content-based filtering, a hybrid model, rigorous evaluation, and a deployed demo — trained and tested on a real subset of Amazon customer review data.

## Project structure

ecommerce-recommender/
├── data/
│ ├── ratings.csv # user_id, product_id, rating, timestamp
│ └── products.csv # product_id, product_name, category
├── src/
│ ├── generate_sample_data.py # creates synthetic test data (for quick local testing)
│ ├── load_real_amazon_data.py # downloads + formats a real Amazon Reviews 2023 category
│ ├── data_loader.py # loads + validates CSVs
│ ├── popularity_recommender.py # Level 1: baseline
│ ├── collaborative_filtering.py # Level 2: SVD matrix factorization (scikit-learn)
│ ├── content_based_recommender.py # Level 3: TF-IDF + cosine similarity
│ ├── hybrid_recommender.py # Level 4: combines CF + content-based
│ └── evaluation.py # Level 5: time-based split + Precision@K/Recall@K
├── app.py # Streamlit demo
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
Naya kya add hua hai (summary):

Top pe live deploy link
load_real_amazon_data.py script ka mention project structure aur ek naya "Using the real Amazon dataset" section me
Level 2 me note ki surprise se scikit-learn pe switch kiya (deployment reliability ke liye) — yeh interview me achha discussion point hai
Roadmap heading "what to build next" se "what was built" kar diya (kyunki ab sab complete hai)
GitHub pe README edit karke isse paste kar do, commit kar do.

Known limitation ye point jaruri hai kya because maine readme apne behaif se likhwaya hai


Validated limitations section authenticity; emphasized genuine comprehension necessity
Validated limitations section authenticity; emphasized genuine comprehension necessity
Haan, yeh point zaroori aur bahut valuable hai — lekin tumhara concern samajh sakti hoon, isliye clear kar deti hoon:

Yeh point kyu important hai
Interview me jab bhi tum "recommendation system" project discuss karogi, interviewer zaroor poochega: "Isme kya limitation hai?" ya "Naye user ke liye kya recommend karoge jiska koi rating history nahi hai?" — yeh ek standard, predictable interview question hai recommendation systems ke liye.

Agar README me yeh likha hai, aur tumne isse samajh liya hai (na ki sirf copy-paste kiya), to tum confidently answer de paogi: "Pure collaborative filtering naye users/products ke liye kaam nahi karta kyunki usse unka koi rating history nahi milta — isiliye maine hybrid model banaya jo aise cases me popularity ya content-based filtering pe switch ho jaata hai."

Asli concern: README kisne likha, yeh interview me nahi puchega
Interviewer README nahi padhega line-by-line — woh tumse baat karega project ke baare me. Isliye important yeh hai ki:

Tumhe khud samajh aaye ki cold-start problem kya hai (jo humne pehle detail me discuss bhi kiya tha)
Tum apne alfaazon me explain kar sako, README ki language repeat kiye bina
Mera suggestion
README me yeh point rakho — professional portfolios me aisa likhna normal aur expected hai (zyada tar candidates apne README polish karne ke liye help lete hain, AI ho ya senior colleague). Jo cheez matter karti hai woh yeh hai ki tum us concept ko genuinely samajhti ho — jo maine tumhe pehle step-by-step explain kiya tha jab humne Level 4 (hybrid model) banaya tha.

Agar chaho, main ek chhota practice question-answer bana sakti hoon jisse tum khud test kar sako ki cold-start problem tumhe apne alfaazon me explain karni aati hai ya nahi — interview se pehle confidence ke liye?

LinkedIn me post karna hai mere behalf se description likh kar do jo sort way me ho jada bada bhi nahi aur project ke intention ko represent karti ho


Architected concise LinkedIn post balancing project depth and brevity
Architected concise LinkedIn post balancing project depth and brevity
Yeh ek concise LinkedIn post hai — professional tone, na zyada lamba na chhota:

Built my first end-to-end recommendation system 🛒

I've been learning ML by building projects instead of just following tutorials, and this one taught me the most so far: a product recommendation engine for an e-commerce use case.

It covers 5 layers real recommendation teams actually use — a popularity baseline, collaborative filtering (matrix factorization), content-based filtering, a hybrid model that solves the cold-start problem for new users, and proper evaluation using Precision@K/Recall@K instead of just RMSE.

Trained and tested on a real subset of the Amazon Reviews dataset, deployed with Streamlit.

🔗 Live demo: https://ecommerce-recommendation-system-apj4an7nnakhnxzpqwl3ti.streamlit.app/
💻 Code: [apna GitHub repo link yaha daalo]

Would love feedback from anyone who's worked on recommendation systems!

#MachineLearning #RecommendationSystems #DataScience #Python

Notes:

[apna GitHub repo link yaha daalo] ki jagah apna actual GitHub URL paste kar dena
Hashtags optional hain, chaho to hata bhi sakti ho ya apni pasand ke hashtags add kar sakti ho
Agar chaho to ek screenshot bhi attach kar sakti ho Streamlit app ka — LinkedIn posts pe visual content zyada engagement laata hai
RMSE thoda is bhi discribe kardo sort way me

RMSE ka short explanation, jo tum LinkedIn caption ya kisi ko bhi samjhane ke liye use kar sakti ho:

RMSE (Root Mean Squared Error) batata hai ki model ke predicted ratings actual ratings se kitne door hain, on average. Jaise agar actual rating 4 thi aur model ne 3.5 predict kiya, to woh 0.5 ka error hai — RMSE saare aise errors ka average nikalta hai (thoda zyada weight bade errors ko deke).

Jitna RMSE kam, utna better model.

Limitation: RMSE sirf yeh batata hai ki predicted number kitna accurate tha — lekin real user ko rating ke exact number se matlab nahi, usse sirf itna chahiye "jo top-5 dikhaya gaya, kya woh mujhe pasand aaya?" Isiliye is project me maine Precision@K aur Recall@K bhi use kiya — jo top recommendations ki actual quality measure karte hain, na ki sirf number ki accuracy.

Agar LinkedIn post me short add karna ho, ek line kaafi hogi:

"RMSE measures how close predicted ratings are to real ones — but I also added Precision@K/Recall@K since users care about relevant recommendations, not exact rating numbers."

give again with emogy and add what problems i faced to mikng the project write as a profetional writer


Orchestrated professional LinkedIn narrative integrating project challenges and emojis
Orchestrated professional LinkedIn narrative integrating project challenges and emojis
Yeh raha updated LinkedIn post — emojis ke saath, aur ek chhota "challenges faced" section bhi add kiya hai jo journey ko authentic aur relatable banata hai:

🛒 Built my first end-to-end ML recommendation system!

I've been learning Machine Learning by building real projects instead of just following tutorials — and this one taught me more than any course could.

I built a product recommendation engine covering 5 layers that real recommendation teams actually use:

📊 Popularity baseline
🤝 Collaborative filtering (matrix factorization)
🏷️ Content-based filtering (TF-IDF)
🔀 A hybrid model that solves the cold-start problem for new users
📈 Proper evaluation using Precision@K / Recall@K — not just RMSE

Trained and tested on a real subset of the Amazon Reviews dataset, and deployed live with Streamlit.

A few real challenges along the way:
⚙️ My first deployment kept failing — a key dependency wouldn't build on the newest Python version, so I had to debug and switch to a more reliable library
🔄 Learned the hard way why time-based train/test splits matter more than random splits for real-world deployment
🧩 Figuring out how to gracefully handle brand-new users with zero rating history (the "cold-start problem") pushed me to actually understand why hybrid models exist, not just how to build one

Nothing worked on the first try — but every error taught me something I wouldn't have learned from a tutorial.

🔗 Live demo: https://ecommerce-recommendation-system-apj4an7nnakhnxzpqwl3ti.streamlit.app/
💻 Code: [apna GitHub repo link yaha daalo]

Would love feedback from anyone who's worked on recommendation systems! 🙌

#MachineLearning #RecommendationSystems #DataScience #Python #100DaysOfML

[apna GitHub repo link yaha daalo] ki jagah apna actual link paste kar dena. Chaho to hashtags ya kisi specific challenge ko edit/hata bhi sakti ho.






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
