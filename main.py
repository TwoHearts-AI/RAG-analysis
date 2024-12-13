from fastapi import FastAPI, UploadFile, File, HTTPException
from sentence_transformers import SentenceTransformer
from config import CONFIG
from generators.MistralGenerator import mistral
from qdrant.QdrantClient import qdrant_client
from chunker.Text_chunker import chunker
from schemas import UploadRequest, UploadResponse, SearchRequest, SearchResponse, RAGRequest, RAGResponse, CollectionListResponse

app = FastAPI()
encoder = SentenceTransformer(CONFIG.EMBEDDING_MODEL)


@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    request: UploadRequest
):
    try:
        content = await file.read()
        text = content.decode()
        chunks = chunker.split_text(text)
        embeddings = encoder.encode(chunks)

        qdrant_client.ensure_collection_exists(
            collection_name=request.collection_name,
            vector_size=len(embeddings[0])
        )

        points = [
            {
                "id": i,
                "vector": embedding.tolist(),
                "payload": {"text": chunk}
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        qdrant_client.save_chunks(
            collection_name=request.collection_name,
            chunks=chunks,
            embeddings=embeddings,
            points=points
        )

        return UploadResponse(
            chunks_count=len(chunks),
            collection_name=request.collection_name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    try:
        query_vector = encoder.encode(request.text).tolist()
        results = qdrant_client.search_by_vector(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit
        )

        return SearchResponse(
            results=[{
                "text": hit.payload["text"],
                "score": hit.score
            } for hit in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag", response_model=RAGResponse)
async def rag_inference(request: RAGRequest):
    try:
        query_vector = encoder.encode(request.text).tolist()
        results = qdrant_client.search_by_vector(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit
        )

        context = "\n\n".join([hit.payload["text"] for hit in results])
        response = mistral.generate(
            system_prompt="You are a helpful assistant. Answer based on the provided context.",
            user_query=request.text,
            context=context
        )

        return RAGResponse(
            answer=response,
            sources=[{
                "text": hit.payload["text"],
                "score": hit.score
            } for hit in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    try:
        collections = qdrant_client.client.get_collections()
        stats = []
        for collection in collections.collections:
            count = qdrant_client.client.get_collection(collection.name).vectors_count
            stats.append({
                "name": collection.name,
                "vectors_count": count
            })
        return CollectionListResponse(collections=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
