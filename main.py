from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Furniture Recommender")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load dataset and embeddings
# -------------------------
df = pd.read_csv("data/furniture.csv")
print(f"Loaded {len(df)} products.")

df['description'] = df['description'].fillna("").astype(str)

# Load cached embeddings
embedding_cache_file = "data/embeddings.npy"
df['embedding'] = list(np.load(embedding_cache_file, allow_pickle=True))
print("Loaded cached embeddings!")

# -------------------------
# Load sentence transformer once
# -------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("SentenceTransformer model loaded!")

# -------------------------
# Pydantic request model
# -------------------------
class UserMessage(BaseModel):
    message: str

# -------------------------
# Recommend endpoint
# -------------------------
@app.post("/recommend")
def recommend(msg: UserMessage):
    try:
        query = msg.message

        # Compute query embedding
        query_emb = embed_model.encode(query, convert_to_numpy=True)

        # Compute cosine similarity
        sims = cosine_similarity([query_emb], list(df['embedding']))[0]

        # Top 5 products
        top_indices = np.argsort(sims)[-5:][::-1]
        top_products = df.iloc[top_indices][['uniq_id', 'title', 'description', 'images']].to_dict(orient='records')

        # Format for frontend
        result = []
        for p in top_products:
            img_url = "https://via.placeholder.com/200x150"  # default
            if pd.notna(p['images']):
                try:
                    images_list = eval(p['images'])
                    if images_list:
                        img_url = images_list[0].strip()
                except:
                    pass

            result.append({
                "id": p['uniq_id'],
                "name": p['title'],
                "img": img_url,
                "description": p['description']
            })

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# Products endpoint
# -------------------------
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
