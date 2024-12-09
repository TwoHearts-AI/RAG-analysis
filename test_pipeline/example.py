import os
from mistralai import Mistral
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from Qdrant.QdrantClient import QdrantClient
from Generators.MistralGenerator import MistralLLM
from Retriever.QdrantRetriever import QdrantRetriever
from Generators.promts.promt import rag_search_prompts, llm_inference_prompts
# Инициализация Qdrant клиента
qdrant = QdrantClient()

# Инициализация SentenceTransformer
model = SentenceTransformer("deepvk/USER-bge-m3", device='cpu')

# Инициализация компонентов
retriever = QdrantRetriever(
    collection_name="chat_chunks_baseline_default",
    model=model,
    qdrant_client=qdrant,
    limit=5
)

llm = MistralLLM()


def rag_pipeline(rag_search_prompts, query, retriever, llm):

    relevant_context = ''
    for search in rag_search_prompts:
        relevant_context += retriever.search_relevant_context(search)

    print("Релевантный контекст:", query)

    inference_prompt = f"Контекст:\n{relevant_context}\n\nВопрос:\n{query}\n\nОтвет:"
    answer = llm(inference_prompt)
    return answer


rag_search_prompts = rag_search_prompts
query = llm_inference_prompts[0]

response = rag_pipeline(rag_search_prompts, query, retriever, llm)

print(query)