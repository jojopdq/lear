from typing import List

import httpx

from lear.rag_provider import RAGProvider


class HttpRagProvider(RAGProvider):
    def __init__(self, mode: str, url: str, access_token: str):
        super().__init__(mode)
        self.resp = None
        self.url = url
        self.access_token = access_token

    def ask(self, question: str):
        headers = {'Authorization': f'Bearer {self.access_token}'}
        url = f'{self.url}/ask?mode={self.name}&question={question}'
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
