from mistralai import Mistral
from mistralai import Mistral
from config import CONFIG


class MistralGenerator:
    def __init__(self):
        self.client = Mistral(api_key=CONFIG.MISTRAL_API_KEY)
        self.model = CONFIG.MISTRAL_MODEL

    def generate(self, system_prompt: str, user_query: str, context: str = None) -> str:
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


mistral = MistralGenerator()
