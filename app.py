from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI()

# Ücretsiz ve basit model (şimdilik)
HF_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
HF_KEY = os.getenv("HF_API_KEY", "")

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(q: Question):
    headers = {
        "Authorization": f"Bearer {HF_KEY}"
    }

    payload = {
        "inputs": q.question,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.4
        }
    }

    r = requests.post(HF_API, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        return {"error": "Model cevap vermedi"}

    data = r.json()

    # HuggingFace bazen liste döndürür
    if isinstance(data, list) and len(data):
        return {"answer": data[0]["generated_text"]}

    return {"answer": data}
