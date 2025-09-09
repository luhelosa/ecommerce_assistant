from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, re, requests

REC_API_BASE = os.environ.get("REC_API_BASE", "http://localhost:8000")

app = FastAPI(title="Mini Bot e-commerce")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Msg(BaseModel):
    message: str
    user_id: int | None = 1

@app.post("/reply")
def reply(msg: Msg):
    text = msg.message.strip()
    try:
        # intents simples por regex
        if re.search(r"recom(i|ie)nd", text, re.I):
            k = 5
            r = requests.get(f"{REC_API_BASE}/recommend/user/{msg.user_id}?k={k}", timeout=5)
            items = r.json().get("items", [])
            if not items:
                return { "reply": "Sin recomendaciones por ahora." }
            lines = ["Recomendaciones:"] + [f"- ({it['product_id']}) {it['title']} – ${it['price']}" for it in items]
            return {"reply": "\n".join(lines)}

        m = re.search(r"similares?\s+al?\s+(\d+)", text, re.I)
        if m:
            pid = int(m.group(1))
            r = requests.get(f"{REC_API_BASE}/recommend/similar/{pid}?k=5", timeout=5)
            items = r.json().get("items", [])
            if not items:
                return { "reply": "No encontré similares." }
            lines = [f"Similares a {pid}:"] + [f"- ({it['product_id']}) {it['title']} – ${it['price']}" for it in items]
            return {"reply": "\n".join(lines)}

        m = re.search(r"(info|informaci[oó]n|detalle).*(\d+)", text, re.I)
        if m:
            pid = int(m.group(2))
            r = requests.get(f"{REC_API_BASE}/products/{pid}", timeout=5)
            it = r.json()
            return {"reply": f"{it['title']} – ${it['price']}\n{it['description']}"}

        # fallback
        return {"reply": "No te entendí. Prueba: 'recomiéndame productos', 'productos similares al 1' o 'información del producto 3'."}
    except Exception as e:
        return {"reply": f"Tu API no respondió ({e}). ¿Está corriendo en {REC_API_BASE}?"}