from qdrant_client import QdrantClient as SyncQdrantClient
from qdrant_client.http import models
from qdrant_client.models import ScoredPoint
from loguru import logger
from typing import List
from langsmith import traceable
import math

from config import CONFIG


class QdrantClient:
    def __init__(self):
        self.client = SyncQdrantClient(
            url=CONFIG.QDRANT_URL,
            https=False,
            port=None,
        )
        self.batch_size = 20

    def ensure_collection_exists(
            self,
            collection_name: str,
            vector_size: int
    ) -> None:
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name not in collection_names:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")

    @traceable
    def save_chunks(
            self,
            collection_name: str,
            chunks: List[str],
            vectors: List[List[float]],
            filename: str,

    ) -> None:
        self.ensure_collection_exists(
            collection_name=collection_name,
            vector_size=len(vectors[0])
        )

        total_chunks = len(chunks)
        num_batches = math.ceil(total_chunks / self.batch_size)

        for batch_num in range(num_batches):
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, total_chunks)

            batch_points = []
            for i, (chunk, vector) in enumerate(zip(
                    chunks[start_idx:end_idx],
                    vectors[start_idx:end_idx]
            )):
                metadata = {
                    "filename": filename,
                }
                point_id = start_idx + i
                batch_points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"content": chunk,  "metadata": metadata, }
                ))

            self.client.upsert(
                collection_name=collection_name,
                points=batch_points
            )

            logger.info(
                f"Saved batch {batch_num + 1}/{num_batches} "
                f"({len(batch_points)} chunks) in collection {collection_name}"
            )

    @traceable
    def search_by_vector(
            self,
            collection_name: str,
            query_vector: List[float],
            limit: int = 10
    ) -> List[ScoredPoint]:
        """Search vectors with basic filtering"""
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results


qdrant_client = QdrantClient()
