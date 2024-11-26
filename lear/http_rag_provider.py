from typing import List

import httpx

from lear.rag_provider import RAGProvider


class HttpRagProvider(RAGProvider):
    def __init__(self, mode: str):
        super().__init__(mode)
        self.resp = None

    def ask(self, question: str):
        headers = {'Authorization': 'Bearer abc123'}
        url = f'http://305-server:5566/ask?mode={self.name}&question={question}'
        resp = httpx.get(url, headers=headers, timeout=90)
        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(resp.text)

    def retrieve(self, query: str, metadata: dict) -> List[str]:
        self.resp = self.ask(query)
        return self.resp.get('retrieved_contexts', [])

    def generate(self, query: str, metadata: dict, retrieved_contexts: List[str]) -> str:
        return self.resp.get('answer')
