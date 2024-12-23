from sentence_transformers import SentenceTransformer
from Qdrant.QdrantClient import QdrantClient
from Generators.MistralClient import MistralClient
from prompts.llm_inference import system_prompt, llm_query_prompt
from prompts.vector_search import vector_search_prompts
from reranker.Reranker import Reranker


qdrant = QdrantClient()
mistral = MistralClient()



def rag_pipeline(vector_search_prompts, query, mistral, qdrant, collection_name)->str:
    merged_prompt_text = ''

    relevant_context = []

    for search_prompt in vector_search_prompts:

        vector = mistral.get_embeddings_batch([search_prompt])[0]

        mass = []

        for i in qdrant.search_by_vector(collection_name=collection_name, query_vector=vector, limit=5):
            mass.append(i.payload['content']['content'])

        relevant_context.append(mass)

    reranker = Reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_list = reranker.rerank(vector_search_prompts, relevant_context)

    print("Релевантный контекст:", reranker_list)
    return 0
    answer = mistral.inference_llm(system_prompt, query, reranker_list)

    return answer

rag_search_prompts = vector_search_prompts
query = llm_query_prompt

response = rag_pipeline(rag_search_prompts, query, mistral, qdrant, "chat_chunks_baseline_default")

print(response)
# limit = 10
# cnt_query = len(results) / 10
# candidates = []
# for i in range(cnt_query):
#     candidates.append([
#         (query, result)  # Преобразуем в строку
#         for result in results[10 * i:10 + 10 * i]
#     ])
# print(candidates)