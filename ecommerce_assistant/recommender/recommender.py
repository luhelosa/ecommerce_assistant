import os
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import joblib

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")

_products = None
_vec = None
_X = None   # TF-IDF matriz de productos (csr)
_UI = None  # user-item (csr)
_uids = None
_pop = None

def _load_once():
    global _products, _vec, _X, _UI, _uids, _pop
    if _products is None:
        _products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    if _pop is None:
        pop_path = os.path.join(MODELS_DIR, "popularity.csv")
        if os.path.exists(pop_path):
            _pop = pd.read_csv(pop_path)
        else:
            # fallback simple: popularidad por frecuencia en interactions si existiera, si no, por precio
            _pop = _products[["product_id"]].copy()
            _pop["score"] = 1.0
    # Modelos
    tfidf_path = os.path.join(MODELS_DIR, "product_tfidf.npz")
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    ui_path = os.path.join(MODELS_DIR, "user_item.npz")
    uids_path = os.path.join(MODELS_DIR, "user_ids.csv")

    if _X is None and os.path.exists(tfidf_path):
        _X = sparse.load_npz(tfidf_path).tocsr()
        _X = normalize(_X)
    if _vec is None and os.path.exists(vec_path):
        _vec = joblib.load(vec_path)
    if _UI is None and os.path.exists(ui_path):
        _UI = sparse.load_npz(ui_path).tocsr()
    if _uids is None and os.path.exists(uids_path):
        _uids = pd.read_csv(uids_path)["user_id"].values

def recommend_for_user(user_id: int, k: int = 5):
    _load_once()
    # Fallback a popularidad si no hay UI o user desconocido
    if _UI is None or _uids is None or user_id not in set(_uids):
        top = _pop.sort_values("score", ascending=False).head(k)
        return _products[_products["product_id"].isin(top["product_id"])].to_dict(orient="records")

    # items consumidos por el usuario
    uid_index_map = {u:i for i, u in enumerate(_uids)}
    uidx = uid_index_map[user_id]
    row = _UI.getrow(uidx)  # vector 1 x n_items
    consumed = set(row.indices.tolist())  # índices de columna (0-based product index)

    # Recomendación simple: productos populares que NO estén consumidos
    top = _pop.sort_values("score", ascending=False)["product_id"].tolist()
    rec_ids = []
    for pid in top:
        if (pid - 1) not in consumed:
            rec_ids.append(pid)
        if len(rec_ids) >= k:
            break
    # Si aún faltan, completar con los primeros productos
    if len(rec_ids) < k:
        for pid in _products["product_id"].tolist():
            if (pid - 1) not in consumed and pid not in rec_ids:
                rec_ids.append(pid)
            if len(rec_ids) >= k:
                break
    return _products[_products["product_id"].isin(rec_ids)].head(k).to_dict(orient="records")

def similar_items(product_id: int, k: int = 5):
    _load_once()
    if _X is None:
        # Fallback: devolver los siguientes k productos
        start = max(int(product_id) - 1, 0)
        ids = _products["product_id"].tolist()
        rec = [pid for pid in ids if pid != int(product_id)]
        return _products[_products["product_id"].isin(rec)].head(k).to_dict(orient="records")

    idx = int(product_id) - 1
    if idx < 0 or idx >= _X.shape[0]:
        return []
    v = _X.getrow(idx)
    sims = cosine_similarity(v, _X).ravel()
    order = np.argsort(-sims)
    rec_idx = [i for i in order if i != idx][:k]
    rec_ids = [int(i + 1) for i in rec_idx]
    return _products[_products["product_id"].isin(rec_ids)].to_dict(orient="records")

def search_products(q: str, k: int = 10):
    _load_once()
    # Si hay vectorizador, usar TF-IDF; si no, filtro simple
    if _vec is not None and _X is not None:
        qv = _vec.transform([q])
        qv = normalize(qv)
        sims = cosine_similarity(qv, _X).ravel()
        order = np.argsort(-sims)[:k]
        rec_ids = [int(i + 1) for i in order]
        return _products[_products["product_id"].isin(rec_ids)].to_dict(orient="records")
    # Filtro simple por columnas
    m = (
        _products["title"].str.contains(q, case=False, na=False) |
        _products["category"].str.contains(q, case=False, na=False) |
        _products["tags"].str.contains(q, case=False, na=False) |
        _products["description"].str.contains(q, case=False, na=False)
    )
    return _products[m].head(k).to_dict(orient="records")