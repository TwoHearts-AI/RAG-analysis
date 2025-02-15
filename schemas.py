from pydantic import BaseModel, Field
from typing import Any, List


class SearchResult(BaseModel):
    text: Any
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


class SearchRequest(BaseModel):
    text: str
    collection_name: str = Field(default="default_collection")
    limit: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    results: List[SearchResult]


class RAGRequest(BaseModel):
    collection_name: str = Field(default="default_collection")
    limit: int = Field(default=3, ge=1, le=10)


class RAGResponse(BaseModel):
    answer: str
    context: str


class CollectionStats(BaseModel):
    name: str
    vectors_count: int


class CollectionListResponse(BaseModel):
    collections: List[CollectionStats]


class UploadRequest(BaseModel):
    collection_name: str = Field(
        default="default_collection",
        description="Name of the collection to upload to"
    )


class UploadResponse(BaseModel):
    chunks_count: int
    collection_name: str
    message: str = Field(default="Upload successful")
