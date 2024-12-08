from qdrant_client import QdrantClient as SyncQdrantClient
from qdrant_client.http import models
from loguru import logger
from uuid import UUID
import uuid
from typing import List, Dict, Optional
import math
from qdrant_client.models import ScoredPoint
import os

class QdrantClient:
    def __init__(self):
        self.client = SyncQdrantClient(
            url=os.getenv('SERVER'),
            https=False,
            port=None,
        )
        self.batch_size = 10  # Default batch size for uploads

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

    def save_chunks(
            self,
            collection_name: str,
            chunks: List[Dict],
            vectors: List[List[float]],
            document_id: UUID,
            chat_id: UUID,
            filename: str,
    ) -> None:
        self.ensure_collection_exists(
            collection_name=collection_name,
            vector_size=len(vectors[0])
        )

        total_chunks = len(chunks)
        num_batches = math.ceil(total_chunks / self.batch_size)

        for batch_num in range(num_batches)[281:]:
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, total_chunks)

            batch_points = []
            for i, (chunk, vector) in enumerate(zip(
                    chunks[start_idx:end_idx],
                    vectors[start_idx:end_idx]
            )):
                metadata = {
                    "document_id": str(document_id),
                    "chat_id": str(chat_id),
                    "filename": filename,
                    "chunk_index": start_idx + i
                }

                point_id = str(uuid.uuid4())

                batch_points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "metadata": metadata,
                        "content": chunk
                    }
                ))

            self.client.upsert(
                collection_name=collection_name,
                points=batch_points
            )

            logger.info(
                f"Saved batch {batch_num + 1}/{num_batches} "
                f"({len(batch_points)} chunks) in collection {collection_name}"
            )

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
