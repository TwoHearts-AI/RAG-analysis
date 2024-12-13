from fastapi import FastAPI, UploadFile, File, HTTPException, status
from sentence_transformers import SentenceTransformer
from config import CONFIG
from generators.MistralGenerator import mistral
from qdrant.QdrantClient import qdrant_client
from chunker.Text_chunker import chunker
from schemas import SearchResult, UploadRequest, UploadResponse, SearchRequest, SearchResponse, RAGRequest, RAGResponse, CollectionListResponse
import uuid
app = FastAPI()
encoder = SentenceTransformer(CONFIG.EMBEDDING_MODEL)


# main.py
@app.post(
    "/upload/{collection_name}",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED
)
async def upload_file(
    collection_name: str,
    file: UploadFile = File(...),

):
    try:
        content = await file.read()
        text = content.decode()
        chunks = chunker.split_text(text)
        embeddings = encoder.encode(chunks)

        document_id = uuid.uuid4()
        chat_id = uuid.uuid4()

        qdrant_client.ensure_collection_exists(
            collection_name=collection_name,
            vector_size=len(embeddings[0])
        )

        # Call save_chunks with correct arguments
        qdrant_client.save_chunks(
            collection_name=request.collection_name,
            chunks=chunks,
            vectors=embeddings,  # renamed from embeddings to vectors
            document_id=document_id,
            chat_id=chat_id,
            filename=file.filename or 'unnamed_file'
        )

        return UploadResponse(
            chunks_count=len(chunks),
            collection_name=request.collection_name,
            message="Upload successful"
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

        search_results = [
            SearchResult(
                text=res.payload.get("content", ""),
                score=res.score
            ) for res in results
        ]

        return SearchResponse(results=search_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-inference", response_model=RAGResponse)
async def rag_inference(request: RAGRequest):
    try:
        query_vector = encoder.encode(request.text).tolist()
        results = qdrant_client.search_by_vector(
            collection_name=request.collection_name,
            query_vector=query_vector,
            limit=request.limit
        )

        context = "\n\n".join([res.payload.get("content", "") for res in results])
        response = mistral.generate(
            system_prompt="You are a helpful assistant. Answer based on the provided context.",
            user_query=request.text,
            context=context
        )

        search_results = [
            SearchResult(
                text=res.payload.get("content", ""),
                score=res.score
            ) for res in results
        ]

        return RAGResponse(
            answer=response,
            sources=search_results
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
