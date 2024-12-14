from mistralai import Mistral
from typing import List
from config import CONFIG
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from loguru import logger


class MistralClient:
    def __init__(self):
        self.client = Mistral(api_key=CONFIG.MISTRAL_API_KEY)
        self.model = CONFIG.MISTRAL_MODEL
        self.embed_model = "mistral-embed"
        self.delay = 2.0  # Увеличили до 2 секунд

    @retry(wait=wait_exponential(multiplier=2, min=4, max=20), stop=stop_after_attempt(5))
    def _get_embeddings_with_retry(self, batch: List[str]) -> List[List[float]]:
        """Single batch request with retry logic"""
        response = self.client.embeddings.create(
            model=self.embed_model,
            inputs=batch
        )
        return [data.embedding for data in response.data]

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:  # Уменьшили размер батча
        """Process texts in batches with rate limiting"""
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches")
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            current_batch = i // batch_size + 1
            try:
                batch_embeddings = self._get_embeddings_with_retry(batch)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Batch {current_batch}/{total_batches} processed")
                time.sleep(self.delay)
            except Exception as e:
                logger.info(f"Error in batch {current_batch}: {str(e)}")
                raise

        return all_embeddings

    def inference_llm(self, system_prompt: str, user_query: str, context: str = None) -> str:
        logger.info("Starting LLM inference")
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {user_query}"
        else:
            prompt = user_query

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.complete(
            model=self.model,
            messages=messages
        )
        logger.info("LLM inference completed")
        return response.choices[0].message.content


mistral = MistralClient()
