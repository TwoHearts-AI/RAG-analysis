from sentence_transformers import SentenceTransformer
from Qdrant.QdrantClient import QdrantClient
from Generators.MistralClient import MistralClient
from prompts.llm_inference import system_prompt, llm_query_prompt
from prompts.vector_search import vector_search_prompts
from Reranger.Reranger import Reranker


qdrant = QdrantClient()
mistral = MistralClient()



def rag_pipeline(rag_search_prompts, query, mistral, qdrant)->str:
    merged_prompt_text = ''
    relevant_context = []

    for search_prompt in rag_search_prompts:
        vector = mistral.get_embeddings_batch([search_prompt])[0]

        for i in qdrant.search_by_vector(collection_name="chat_chunks_baseline_default", query_vector=vector, limit=5):
            relevant_context.append(i.payload['content']['content'])
        merged_prompt_text += search_prompt

    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")

    print("Релевантный контекст:", relevant_context)

    reranker_list = reranker.rerank(merged_prompt_text, relevant_context)

    print("Релевантный контекст:", reranker_list)

    answer = mistral.inference_llm(system_prompt, query, reranker_list)

    return answer

rag_search_prompts = vector_search_prompts
query = llm_query_prompt

response = rag_pipeline(rag_search_prompts, query, mistral, qdrant)

print(response)