from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer, util
import torch
import os
import uvicorn

app = FastAPI()

# モデル読み込み（初回のみ少し時間がかかる）
model = SentenceTransformer('all-MiniLM-L6-v2')

@app.post("/similarity")
async def similarity(request: Request):
    data = await request.json()
    s1 = data["text1"]
    s2 = data["text2"]

    e1 = model.encode(s1, convert_to_tensor=True)
    e2 = model.encode(s2, convert_to_tensor=True)

    score = util.pytorch_cos_sim(e1, e2).item()
    return {"similarity": score}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # PORT環境変数を使う（Render用）
    uvicorn.run(app, host="0.0.0.0", port=port)
