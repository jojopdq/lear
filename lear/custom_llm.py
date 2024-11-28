import os

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI


class CustomLlm(DeepEvalBaseLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model(self):
        return OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
            base_url=os.getenv("OPENAI_API_BASE"),
        )

    def generate(self, prompt: str) -> str:
        completion = self.model.chat.completions.create(
            model=self.get_model_name(),
            messages=[
                {"role": "user", "content": prompt}
            ],
            top_p=0.7,
            temperature=0.1
        )
        return completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return os.getenv('OPENAI_API_PROVIDER')
