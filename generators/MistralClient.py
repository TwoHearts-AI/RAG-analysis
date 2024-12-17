from mistralai import Mistral
from typing import List
from config import CONFIG
import time
from loguru import logger
from langsmith import traceable


class MistralClient:
    def __init__(self):
        self.client = Mistral(api_key=CONFIG.MISTRAL_API_KEY)
        self.model = CONFIG.MISTRAL_MODEL
        self.embed_model = "mistral-embed"
        self.delay = 2

    @traceable()
    def _get_embeddings_single(self, batch: List[str]) -> List[List[float]]:
        """Single batch request with error handling"""
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                inputs=batch
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"API Error details: {str(e)}")
            logger.error(f"Batch size: {len(batch)}")
            raise

    @traceable()
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 23) -> List[List[float]]:
        """Process texts in batches with rate limiting"""
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches")
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            current_batch = i // batch_size + 1
            try:
                batch_embeddings = self._get_embeddings_single(batch)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Batch {current_batch}/{total_batches} processed")
                time.sleep(self.delay)
            except Exception as e:
                logger.error(f"Failed to process batch {current_batch}: {str(e)}")
                raise

        return all_embeddings

    @traceable()
    def inference_llm(self, system_prompt: str, llm_query: str, context: str) -> str:
        logger.info("Starting LLM inference")
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        prompt = f"Контекст из релевантной запросу переписки:\n{context}\n\nЗапрос на психологическую консультацию: {llm_query}"

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages
            )
            logger.info("LLM inference completed")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Chat completion error: {str(e)}")
            raise


mistral = MistralClient()
