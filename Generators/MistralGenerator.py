from .generator import GeneratorABC
from mistralai import Mistral
from .promts.promt import general_prompt
import os
class MistralLLM(GeneratorABC):
    def __init__(self):
        self.api_key = os.getenv('MISTRAL_SECRET')
        self.model_name = os.getenv('MISTRAL_LARGE_LATEST')
        self.client = Mistral(api_key=self.api_key)

    def __call__(self, query):
        query = [
            {"role": "system", "content": general_prompt},
            {"role": "user", "content": query}
        ]

        response = self.client.chat.complete(
            model=self.model_name,
            messages=query,
        )
        return response.choices[0].message.content