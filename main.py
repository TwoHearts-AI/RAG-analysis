from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from generators.MistralClient import mistral
from qdrant.QdrantClient import qdrant_client
from chunker.Text_chunker import chunker
from schemas import SearchResult, UploadResponse, SearchRequest, SearchResponse, RAGRequest, RAGResponse, CollectionListResponse
from loguru import logger
from prompts.vector_search import vector_search_prompts
from qdrant_client.models import ScoredPoint
from prompts.llm_inference import llm_query_prompt, system_prompt

app = FastAPI()

@app.post("/upload/{collection_name}", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(collection_name: str, file: UploadFile = File(...)):
    try:
        logger.info(f"Starting file upload to collection: {collection_name}")
        content = await file.read()
        text = content.decode()

        logger.info("Splitting text into chunks")
        chunks = chunker.split_text(text)
        logger.info(f"Generated {len(chunks)} chunks")

        logger.info("Generating embeddings")
        embeddings = mistral.get_embeddings_batch(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")

        logger.info("Ensuring collection exists")
        qdrant_client.ensure_collection_exists(
            collection_name=collection_name,
            vector_size=len(embeddings[0])
        )

        logger.info("Saving chunks to Qdrant")
        qdrant_client.save_chunks(
            collection_name=collection_name,
            chunks=chunks,
            vectors=embeddings,
            filename=file.filename or 'unnamed_file'
        )
        logger.info("Upload completed successfully")

        return UploadResponse(
            chunks_count=len(chunks),
            collection_name=collection_name,
            message="Upload successful"
        )
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    try:
        logger.info(f"Searching in collection: {request.collection_name}")
        logger.info("Generating embedding for search query")
        embeddings = mistral.get_embeddings_batch([request.text])[0]

        logger.info(f"Searching for similar vectors, limit: {request.limit}")
        results = qdrant_client.search_by_vector(
            collection_name=request.collection_name,
            query_vector=embeddings,
            limit=request.limit
        )
        logger.info(f"Found {len(results)} results")

        return SearchResponse(
            results=[SearchResult(text=res.payload.get("content", ""), score=res.score)
                     for res in results]
        )
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag-inference", response_model=RAGResponse)
async def rag_inference(request: RAGRequest):
    try:
        logger.info(f"Starting RAG inference for collection: {request.collection_name}")

        # Get embeddings for all search prompts
        logger.info("Generating embeddings for search prompts")
        vector_search_embedding = mistral.get_embeddings_batch(vector_search_prompts)

        # Get results for each search prompt
        vector_search_res: List[ScoredPoint] = []
        for idx, embedding in enumerate(vector_search_embedding):
            logger.info(f"Searching with prompt {idx + 1}")
            results = qdrant_client.search_by_vector(
                collection_name=request.collection_name,
                query_vector=embedding,
                limit=request.limit
            )
            vector_search_res.extend(results)

        # Remove duplicates and sort by score
        unique_results = {}
        for res in vector_search_res:
            content = res.payload.get("content", "")
            if content not in unique_results or res.score > unique_results[content].score:
                unique_results[content] = res

        # Take top N results
        top_results = sorted(
            unique_results.values(),
            key=lambda x: x.score,
            reverse=True
        )[:request.limit]

        logger.info(f"Found {len(top_results)} unique relevant chunks")

        reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

        

        reranker_list = reranker.rerank(merged_prompt_text, relevant_context)

        print("Релевантный контекст:", reranker_list)

        answer = mistral.inference_llm(system_prompt, query, reranker_list)

        # Combine context and generate response
        context = "\n\n".join([res.payload.get("content", "") for res in top_results])
        logger.info("Generating LLM response")

        response = mistral.inference_llm(
            system_prompt=system_prompt,
            llm_query=llm_query_prompt,
            context=context
        )
        logger.info("RAG inference completed")

        return RAGResponse(
            answer=response,
            sources=[SearchResult(text=res.payload.get("content", ""), score=res.score)
                     for res in top_results]
        )
    except Exception as e:
        logger.error(f"Error during RAG inference: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    try:
        logger.info("Listing collections")
        collections = qdrant_client.client.get_collections()
        stats = [
            {
                "name": collection.name,
                "vectors_count": qdrant_client.client.get_collection(collection.name).vectors_count
            }
            for collection in collections.collections
        ]
        logger.info(f"Found {len(stats)} collections")
        return CollectionListResponse(collections=stats)
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
