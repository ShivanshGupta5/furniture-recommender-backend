from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

df = pd.read_csv("data/furniture.csv")
df['description'] = df['description'].fillna("").astype(str)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['description'], convert_to_numpy=True)
np.save("data/embeddings.npy", embeddings)
