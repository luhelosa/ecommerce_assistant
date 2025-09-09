import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def build():
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    inter = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    
    # TF-IDF de contenido (título + categoría + tags + descripción)
    corpus = (
        products["title"].fillna("") + " " +
        products["category"].fillna("") + " " +
        products["tags"].fillna("") + " " +
        products["description"].fillna("")
    )
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(corpus)
    X = normalize(X)
    
    # Matriz usuario-item (pondera eventos)
    wmap = {"view": 1, "add_to_cart": 3, "purchase": 5}
    inter["w"] = inter["event_type"].map(wmap).fillna(0)
    agg = inter.groupby(["user_id", "product_id"])["w"].sum().reset_index()
    
    users = agg["user_id"].astype(int).values
    items = agg["product_id"].astype(int).values
    data = agg["w"].astype(float).values
    
    uids = np.unique(users)
    u2i = {u: i for i, u in enumerate(uids)}
    rows = np.array([u2i[u] for u in users])
    cols = items - 1
    
    UI = sparse.coo_matrix((data, (rows, cols)), shape=(len(uids), X.shape[0])).tocsr()
    
    # Popularidad (fallback)
    pop = inter.groupby("product_id")["w"].sum().sort_values(ascending=False)
    
    # Guardar artefactos
    joblib.dump(vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    sparse.save_npz(os.path.join(MODELS_DIR, "product_tfidf.npz"), X)
    sparse.save_npz(os.path.join(MODELS_DIR, "user_item.npz"), UI)
    pd.Series(uids, name="user_id").to_csv(os.path.join(MODELS_DIR, "user_ids.csv"), index=False)
    pop.to_csv(os.path.join(MODELS_DIR, "popularity.csv"), header=["score"])
    
    print("ok")

if __name__ == "__main__":
    build()
