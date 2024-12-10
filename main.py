from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient
from pydantic import BaseModel, Field, conint
from typing import List, Optional
import os
from loguru import logger
from schemas import SimilarDocsRequest, SimilarDocsResponse, RAGResponse, RAGRequest, CollectionStats, CollectionStatsRequest

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global qdrant_client
    qdrant_client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )


@app.post("/similar-documents", response_model=SimilarDocsResponse)
async def get_similar_documents(request: SimilarDocsRequest):
    """
    Retrieve similar documents from Qdrant without LLM processing
    """
    try:
        # Здесь будет логика получения эмбеддингов для query_text
        # и поиск похожих документов в Qdrant
        
        search_result = qdrant_client.search(
            collection_name=request.collection_name,
            query_vector=[0.1] * 1536,  # Замените на реальные эмбеддинги
            limit=request.limit,
            score_threshold=request.score_threshold
        )
        
        documents = [
            DocumentResponse(
                text=hit.payload.get("text", ""),
                metadata=hit.payload.get("metadata", {}),
                score=hit.score
            ) for hit in search_result
        ]
        
        return SimilarDocsResponse(
            documents=documents,
            total_found=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error in similar-documents endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag-inference", response_model=RAGResponse)
async def run_rag_inference(request: RAGRequest):
    """
    Run full RAG pipeline including LLM inference
    """
    try:
        # 1. Получение эмбеддингов для query_text
        # 2. Поиск похожих документов в Qdrant
        # 3. Подготовка промпта с найденными документами
        # 4. Вызов LLM
        # Это заглушка, нужно реализовать реальную логику
        
        similar_docs = qdrant_client.search(
            collection_name=request.collection_name,
            query_vector=[0.1] * 1536,  # Замените на реальные эмбеддинги
            limit=request.limit
        )
        
        documents = [
            DocumentResponse(
                text=hit.payload.get("text", ""),
                metadata=hit.payload.get("metadata", {}),
                score=hit.score
            ) for hit in similar_docs
        ]
        
        # Здесь будет вызов LLM
        
        return RAGResponse(
            answer="This is a placeholder answer from LLM",
            used_documents=documents,
            total_tokens=100  # Замените на реальное количество токенов
        )
        
    except Exception as e:
        logger.error(f"Error in rag-inference endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    """
    Get detailed statistics about a specific collection
    """
    try:
        # Получение информации о коллекции
        collection_info = qdrant_client.get_collection(request.collection_name)
        
        # Получение количества точек в коллекции
        points_count = qdrant_client.count(
            collection_name=request.collection_name
        ).count
        
        return CollectionStats(
            total_documents=points_count,
            vectors_size=collection_info.config.params.vectors.size,
            collection_name=request.collection_name,
            metadata_schema=collection_info.payload_schema
        )
        
    except Exception as e:
        logger.error(f"Error in collection-stats endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))