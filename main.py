from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
df = pd.read_csv("data/furniture.csv")
print(f"Loaded {len(df)} products.")

# Fill missing descriptions and convert to string
df['description'] = df['description'].fillna("").astype(str)

# Load cached embeddings
embedding_cache_file = "data/embeddings.npy"
df['embedding'] = list(np.load(embedding_cache_file, allow_pickle=True))
print("Loaded cached embeddings!")

# Pydantic model for request
class UserMessage(BaseModel):
    message: str

@app.post("/recommend")
def recommend(msg: UserMessage):
    query = msg.message

    # Use cosine similarity with cached embeddings
    # If you want, you can compute query embeddings with SentenceTransformer here
    # For now, let's assume you still compute query embedding dynamically
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = embed_model.encode(query, convert_to_numpy=True)

    sims = cosine_similarity([query_emb], list(df['embedding']))[0]

    # Top 5 products
    top_indices = np.argsort(sims)[-5:][::-1]
    top_products = df.iloc[top_indices][['uniq_id', 'title', 'description', 'images']].to_dict(orient='records')

    # Format for frontend
    result = []
    for p in top_products:
        if pd.notna(p['images']):
            try:
                images_list = eval(p['images'])
                img_url = images_list[0].strip() if images_list else "https://via.placeholder.com/200x150"
            except:
                img_url = "https://via.placeholder.com/200x150"
        else:
            img_url = "https://via.placeholder.com/200x150"

        result.append({
            "id": p['uniq_id'],
            "name": p['title'],
            "img": img_url,
            "description": p['description']
        })

    return result

@app.get("/products")
def get_products():
    desired_cols = [
        "uniq_id", "title", "brand", "description", "price",
        "categories", "images", "manufacturer",
        "package dimensions", "country_of_origin",
        "material", "color",
    ]
    cols_to_include = [col for col in desired_cols if col in df.columns]
    data = df[cols_to_include].fillna("").to_dict(orient="records")
    return JSONResponse(content=data)
