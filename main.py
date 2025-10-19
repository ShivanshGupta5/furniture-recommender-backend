from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ast
import os

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

# Fill missing descriptions
df['description'] = df['description'].fillna("").astype(str)

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="./models")

# Load precomputed embeddings if exists, otherwise compute and save
embedding_file = "data/embeddings.npy"
if os.path.exists(embedding_file):
    df['embedding'] = list(np.load(embedding_file, allow_pickle=True))
    print("Loaded precomputed embeddings.")
else:
    print("Generating embeddings for products...")
    df['embedding'] = list(embed_model.encode(df['description'], convert_to_numpy=True))
    np.save(embedding_file, df['embedding'])
    print("Embeddings generated and saved!")

# Pydantic model for request
class UserMessage(BaseModel):
    message: str

@app.post("/recommend")
def recommend(msg: UserMessage):
    query = msg.message
    query_emb = embed_model.encode(query, convert_to_numpy=True)
    
    # Compute cosine similarity
    sims = cosine_similarity([query_emb], list(df['embedding']))[0]
    
    # Top 5 products
    top_indices = np.argsort(sims)[-5:][::-1]
    top_products = df.iloc[top_indices][['uniq_id', 'title', 'description', 'images']].to_dict(orient='records')
    
    result = []
    for p in top_products:
        # Extract first valid image from list
        img_url = "https://via.placeholder.com/200x150"
        if pd.notna(p['images']):
            try:
                images_list = ast.literal_eval(p['images'])
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
    return result

@app.get("/products")
def get_products():
    desired_cols = [
        "uniq_id", "title", "brand", "description", "price",
        "categories", "images", "manufacturer", "package dimensions",
        "country_of_origin", "material", "color",
    ]
    cols_to_include = [col for col in desired_cols if col in df.columns]
    data = df[cols_to_include].fillna("").to_dict(orient="records")
    return JSONResponse(content=data)
