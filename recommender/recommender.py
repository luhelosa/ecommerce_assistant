import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
import joblib

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")

def load_models():
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    X = sparse.load_npz(os.path.join(MODELS_DIR, "product_tfidf.npz")).tocsr()
    X = normalize(X)
    UI = sparse.load_npz(os.path.join(MODELS_DIR, "user_item.npz")).tocsr()
    user_ids = pd.read_csv(os.path.join(MODELS_DIR, "user_ids.csv"))["user_id"].astype(int).tolist()
    pop = pd.read_csv(os.path.join(MODELS_DIR, "popularity.csv"))
    pop.index = pop.index.astype(int)
    return products, vec, X, UI, user_ids, pop

def recommend_for_user(user_id: int, k: int = 5):
    products, vec, X, UI, user_ids, pop = load_models()
    try:
        row = user_ids.index(int(user_id))
    except ValueError:
        top_ids = pop.sort_values("score", ascending=False).index.values[:k]
        return products[products["product_id"].isin(top_ids)].head(k).to_dict(orient="records")

    ui = UI[row]
    if ui.nnz == 0:
        top_ids = pop.sort_values("score", ascending=False).index.values[:k]
        return products[products["product_id"].isin(top_ids)].head(k).to_dict(orient="records")

    prof = ui @ X
    denom = (np.linalg.norm(prof.data) + 1e-12) if hasattr(prof, "data") else 1.0
    prof = prof / denom

    sims = (X @ prof.T).toarray().ravel()
    seen = set(ui.indices + 1)
    order = np.argsort(-sims)

    rec = []
    for i in order:
        pid = int(i + 1)
        if pid not in seen:
            rec.append(pid)
        if len(rec) >= k:
            break

    return products[products["product_id"].isin(rec)].head(k).to_dict(orient="records")

def similar_items(product_id: int, k: int = 5):
    products, vec, X, UI, user_ids, pop = load_models()
    idx = int(product_id) - 1
    if idx < 0 or idx >= X.shape[0]:
        return []

    v = X[idx]
    sims = (X @ v.T).toarray().ravel()
    order = np.argsort(-sims)
    rec = []
    
    for j in order:
        pid = int(j + 1)
        if pid != int(product_id):
            rec.append(pid)
        if len(rec) >= k:
            break
    
    return products[products["product_id"].isin(rec)].head(k).to_dict(orient="records")


def search_products(q: str, k: int = 10):
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    m = (
        products["title"].str.contains(q, case=False, na=False) | 
        products["category"].str.contains(q, case=False, na=False) | 
        products["tags"].str.contains(q, case=False, na=False) | 
        products["description"].str.contains(q, case=False, na=False)
    )
    
    return products[m].head(k).to_dict(orient="records")

