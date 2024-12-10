from pydantic import BaseModel, Field, conint
from typing import List, Optional

class SimilarDocsRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the Qdrant collection")
    query_text: str = Field(..., description="Text to search similar documents for")
    limit: conint(gt=0, le=20) = Field(5, description="Number of documents to return")
    score_threshold: Optional[float] = Field(0.3, description="Minimum similarity score threshold")

class DocumentResponse(BaseModel):
    text: str
    metadata: dict
    score: float

class SimilarDocsResponse(BaseModel):
    documents: List[DocumentResponse]

class RAGRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the Qdrant collection")
    query_text: str = Field(..., description="User query for RAG processing")
    limit: conint(gt=0, le=10) = Field(3, description="Number of documents to retrieve")

class RAGResponse(BaseModel):
    answer: str
    used_documents: List[DocumentResponse]

class CollectionStatsRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the Qdrant collection")

class CollectionStats(BaseModel):
    total_documents: int
    vectors_size: int
    collection_name: str
