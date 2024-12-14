from mistralai import Mistral
from typing import List
from config import CONFIG


class MistralClient:
    def __init__(self):
        self.client = Mistral(api_key=CONFIG.MISTRAL_API_KEY)
        self.model = CONFIG.MISTRAL_MODEL
        self.embed_model = "mistral-embed"

    def inference_llm(self, system_prompt: str, user_query: str, context: str = None) -> str:
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
        return response.choices[0].message.content

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Process texts in batches of specified size and return embeddings."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.embed_model,
                inputs=batch
            )
            all_embeddings.extend([data.embedding for data in response.data])

        return all_embeddings


mistral = MistralClient()
