# ...existing code...
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import re
from pydantic import BaseModel
from typing import Optional

from recommender.recommender import recommend_for_user, similar_items, search_products

app = FastAPI(title="E-commerce Recs (mínimo)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))


class Msg(BaseModel):
    message: str
    user_id: Optional[int] = 1


def _first_int(text: str) -> Optional[int]:
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def _classify(text: str):
    t = text.lower()
    if any(g in t for g in ["hola", "buenas", "buenos días", "buenas tardes", "buenas noches"]):
        return ("greet", None)
    if "recomend" in t or "suger" in t or "sugerir" in t:
        return ("ask_recommendations", None)
    if "similar" in t or "parecid" in t:
        return ("ask_similar", _first_int(t))
    if "inform" in t or "detalle" in t or "producto" in t or "info" in t:
        return ("ask_product_info", _first_int(t))
    return ("fallback", None)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/products/{product_id}")
def get_product(product_id: int):
    row = products[products["product_id"] == product_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Not found")
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


@app.post("/chat")
def chat(msg: Msg):
    intent, num = _classify(msg.message)

    try:
        if intent == "greet":
            return {"reply": "¡Hola! Soy tu asistente de compras. Puedo recomendarte productos, mostrar similares o dar info de un producto."}

        if intent == "ask_recommendations":
            k = 5
            try:
                items = recommend_for_user(msg.user_id, k=k)
            except Exception:
                items = []
            if not items:
                # fallback: top-k by product_id (simple fallback si el recomendador falla)
                items = products.head(k).to_dict(orient="records")
            lines = ["Recomendaciones:"]
            lines += [f"* [{it.get('product_id')}] {it.get('title')} - ${it.get('price')}" for it in items]
            return {"reply": "\n".join(lines)}

        if intent == "ask_similar":
            pid = num or 1
            try:
                items = similar_items(pid, k=5)
            except Exception:
                items = []
            if not items:
                return {"reply": f"No encontré productos similares al {pid}."}
            lines = [f"Productos similares a {pid}:"]
            lines += [f"* [{it.get('product_id')}] {it.get('title')} - ${it.get('price')}" for it in items]
            return {"reply": "\n".join(lines)}

        if intent == "ask_product_info":
            pid = num
            if not pid:
                return {"reply": "Dime el id del producto, por ejemplo: 'información del producto 3'."}
            row = products[products["product_id"] == pid]
            if row.empty:
                return {"reply": f"No encontré el producto con id {pid}."}
            p = row.iloc[0].to_dict()
            lines = [
                f"Producto [{p.get('product_id')}]: {p.get('title')}",
                f"Categoría: {p.get('category')}",
                f"Precio: ${p.get('price')}",
                f"Descripción: {p.get('description') or '(sin descripción)'}",
            ]
            return {"reply": "\n".join(lines)}

        # fallback
        return {"reply": "No te entendí. Prueba: 'recomiéndame productos', 'productos similares al 1' o 'información del producto 3'."}

    except Exception as e:
        return {"reply": f"Tu API encontró un error interno ({e})."}
# ...existing