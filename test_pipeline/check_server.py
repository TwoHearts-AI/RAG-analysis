from sentence_transformers import SentenceTransformer
from Qdrant.QdrantClient import QdrantClient

query = 'Как дела?'
qdrant = QdrantClient()
model = SentenceTransformer("deepvk/USER-bge-m3", device='cpu')
vector = model.encode(query, normalize_embeddings=True)

results = qdrant.search_by_vector(
    collection_name="chat_chunks_baseline_default",
    query_vector=vector.tolist(),
    limit=5
)

for i in results:
    print(i.payload['content'])