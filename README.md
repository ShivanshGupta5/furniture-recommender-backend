# 🪑 Furniture Recommender System — Backend

This is the **backend service** for the Furniture Recommender Web App.
It uses **FastAPI** to serve recommendations based on furniture descriptions, powered by **Sentence Transformers** and **Torch embeddings**.

---

## 🚀 Tech Stack

* **FastAPI** — Modern, async web framework for the API
* **Sentence Transformers** — Converts furniture descriptions into semantic embeddings
* **PyTorch** — Used by Sentence Transformers for model computation
* **scikit-learn** — Similarity computation and ranking
* **Uvicorn/Gunicorn** — High-performance ASGI server
* **Render** — Cloud hosting and deployment platform

---

## 🧩 Folder Structure

```
backend/
├── main.py              # FastAPI entry point
├── requirements.txt     # Python dependencies
├── Procfile             # Render process declaration
├── runtime.txt          # Python runtime version
├── model/               # Pretrained model and embeddings
├── data/                # Furniture dataset (optional)
└── README.md
```

---

## ⚙️ Setup (Local Development)

1. **Clone the repo**

   ```bash
   git clone https://github.com/<your-username>/furniture-recommender-backend.git
   cd furniture-recommender-backend
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate       # (Linux/Mac)
   venv\Scripts\activate          # (Windows)
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API locally**

   ```bash
   uvicorn main:app --reload
   ```

5. **Visit in your browser**

   ```
   http://127.0.0.1:8000/docs
   ```

   Swagger UI will open with all available endpoints.

---

## 🌐 Deployment (Render)

1. Push the repo to GitHub.
2. Go to [Render.com](https://render.com).
3. Create a **New Web Service** → Connect your GitHub repo.
4. Under **Start Command**, use:

   ```bash
   gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT
   ```
5. Add a **root route** in `main.py` (optional) to verify deployment:

   ```python
   @app.get("/")
   def root():
       return {"message": "Furniture Recommender API is running!"}
   ```

---

## 🧠 Example Endpoint

### POST `/recommend`

**Request Body:**

```json
{
  "query": "Modern wooden dining table"
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "name": "Oak Dining Set",
      "similarity": 0.89
    },
    {
      "name": "Minimalist Wood Table",
      "similarity": 0.84
    }
  ]
}
```

---

## 🪶 Procfile

```bash
web: uvicorn main:app --bind 0.0.0.0:$PORT
```

---

## 🧰 Requirements (example)

```txt
fastapi==0.115.0
uvicorn==0.30.6
gunicorn==23.0.0
pandas==2.3.3
numpy==2.1.1
scikit-learn==1.5.2
sentence-transformers==3.1.1
torch
```

---

## 📈 Future Enhancements

* Add personalized recommendations based on user profiles
* Integrate database for real product data
* Implement analytics API for user behavior tracking

---

## 👨‍💻 Author

**Shivansh Gupta**
