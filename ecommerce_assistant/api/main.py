from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

# Importar funciones del recomendador de forma segura
try:
    from recommender.recommender import recommend_for_user, similar_items, search_products
except ModuleNotFoundError:
    recommend_for_user = lambda user_id, k=5: []
    similar_items = lambda product_id, k=5: []
    search_products = lambda q, k=10: []
    print("[ADVERTENCIA] No se pudo importar recommender. Usando funciones vacías.")

app = FastAPI(title="E-commerce Recs (mínimo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta segura para el CSV
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
csv_path = os.path.join(DATA_DIR, "products.csv")

if os.path.exists(csv_path):
    products = pd.read_csv(csv_path)
else:
    products = pd.DataFrame()
    print(f"[ADVERTENCIA] No se encontró {csv_path}. La API funcionará sin productos.")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/products/{product_id}")
def get_product(product_id: int):
    if products.empty:
        raise HTTPException(status_code=500, detail="No hay datos de productos")
    row = products[products["product_id"] == product_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    return row.iloc[0].to_dict()

@app.get("/recommend/user/{user_id}")
def rec_user(user_id: int, k: int = 5):
    return {"items": recommend_for_user(user_id, k=k)}

@app.get("/recommend/similar/{product_id}")
def rec_sim(product_id: int, k: int = 5):
    return {"items": similar_items(product_id, k=k)}

@app.get("/search")
def search(q: str = Query(..., min_length=1), k: int = 10):
    return {"items": search_products(q, k=k)}