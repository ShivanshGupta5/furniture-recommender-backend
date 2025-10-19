# Product Recommender (FastAPI + React + Pinecone + LangChain)

## Overview
A product recommendation web app for furniture that:
- returns product recommendations from a vector DB (Pinecone) based on user prompts
- generates creative product descriptions using a lightweight GenAI model (via LangChain)
- includes an analytics page summarizing the dataset
- includes an image classification notebook for the dataset

## Repo structure
(briefly describe)

## Requirements
- Python 3.10+
- Node 18+
- Pinecone account & API key
- (Optional) GPU for training CV models

## Environment variables
Create a `.env` or export these:
- `PINECONE_API_KEY`
- `PINECONE_ENV`
- `PINECONE_INDEX` (default: products-index)
- `EMBED_MODEL_NAME` (default: all-MiniLM-L6-v2)

## Quickstart (local)
1. Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export PINECONE_API_KEY=...
export PINECONE_ENV=...
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
