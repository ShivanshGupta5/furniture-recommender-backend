from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for all product descriptions (precompute for faster queries)
print("Generating embeddings for products...")
df['embedding'] = list(embed_model.encode(df['description'], convert_to_numpy=True))
print("Embeddings ready!")

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
    
    # Format for frontend
    result = []
    for p in top_products:
        # Extract first valid image from list
        if pd.notna(p['images']):
            try:
                # Convert string representation of list to actual Python list
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
            "description": p['description']  # Keep description for tooltips
        })
    return result

@app.get("/products")
def get_products():
    """
    Return all products from the dataset as JSON for analytics or frontend use.
    Only includes columns that exist in the CSV.
    """
    desired_cols = [
        "uniq_id",
        "title",
        "brand",
        "description",
        "price",
        "categories",
        "images",
        "manufacturer",
        "package dimensions",
        "country_of_origin",
        "material",
        "color",
    ]

    # Keep only columns that exist in the DataFrame
    cols_to_include = [col for col in desired_cols if col in df.columns]

    data = df[cols_to_include].fillna("").to_dict(orient="records")
    return JSONResponse(content=data)
