from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI


class CustomLlm(DeepEvalBaseLLM):
    def __init__(self, *args, **kwargs):
        self.model = OpenAI(
            api_key="9e3d5d7d3ff2e74fc9c7e0684c3b6555.feSMhxTcgznEB07f",
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        )

    def load_model(self):
        return self.model

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
        return "glm-4-air"
