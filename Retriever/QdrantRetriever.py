from .retriever import RetrieverABC


class QdrantRetriever(RetrieverABC):
    def __init__(self, collection_name, model, qdrant_client, limit=5):
        super().__init__()
        self.collection_name = collection_name
        self.model = model
        self.limit = limit
        self.qdrant_client = qdrant_client

    def search_relevant_context(self, query):

        vector = self.model.encode(query, normalize_embeddings=True)

        results = self.qdrant_client.search_by_vector(
            collection_name=self.collection_name,
            query_vector=vector.tolist(),
            limit=self.limit
        )
        return "\n".join(str(res.payload['content']) for res in results)