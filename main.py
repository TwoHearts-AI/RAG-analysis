from fastapi import FastAPI, UploadFile, File, HTTPException, status
from generators.MistralClient import mistral
from qdrant.QdrantClient import qdrant_client
from chunker.Text_chunker import chunker
from schemas import SearchResult, UploadResponse, SearchRequest, SearchResponse, RAGRequest, RAGResponse, CollectionListResponse

app = FastAPI()


@app.post("/upload/{collection_name}", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(collection_name: str, file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode()
        chunks = chunker.split_text(text)
        embeddings = mistral.get_embeddings_batch(chunks)

        qdrant_client.ensure_collection_exists(
            collection_name=collection_name,
            vector_size=len(embeddings[0])
        )

        qdrant_client.save_chunks(
            collection_name=collection_name,
            chunks=chunks,
            vectors=embeddings,
            filename=file.filename or 'unnamed_file'
        )

        return UploadResponse(
            chunks_count=len(chunks),
            collection_name=collection_name,
            message="Upload successful"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    try:
        embeddings = mistral.get_embeddings_batch([request.text])[0]
        results = qdrant_client.search_by_vector(
            collection_name=request.collection_name,
            query_vector=embeddings,
            limit=request.limit
        )

        return SearchResponse(
            results=[SearchResult(text=res.payload.get("content", ""), score=res.score)
                     for res in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-inference", response_model=RAGResponse)
async def rag_inference(request: RAGRequest):
    try:
        embeddings = mistral.get_embeddings_batch([request.text])[0]
        results = qdrant_client.search_by_vector(
            collection_name=request.collection_name,
            query_vector=embeddings,
            limit=request.limit
        )

        context = "\n\n".join([res.payload.get("content", "") for res in results])
        response = mistral.inference_llm(
            system_prompt="You are a helpful assistant. Answer based on the provided context.",
            user_query=request.text,
            context=context
        )

        return RAGResponse(
            answer=response,
            sources=[SearchResult(text=res.payload.get("content", ""), score=res.score)
                     for res in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    try:
        collections = qdrant_client.client.get_collections()
        stats = [
            {
                "name": collection.name,
                "vectors_count": qdrant_client.client.get_collection(collection.name).vectors_count
            }
            for collection in collections.collections
        ]
        return CollectionListResponse(collections=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
