from fastapi import FastAPI
from qdrant_client import QdrantClient
import os

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global qdrant_client
    qdrant_client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

@app.get("/")
async def root():
    return {"message": "RAG Analysis API is running"}

@app.get("/status")
async def check_qdrant():
    try:
        collections = qdrant_client.get_collections()
        return {"status": "ok", "collections": collections}
    except Exception as e:
        return {"status": "error", "message": str(e)}