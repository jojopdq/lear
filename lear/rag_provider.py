from typing import List


class RAGProvider(object):
    def __init__(self, name):
        self.name = name

    def retrieve(self, query: str, metadata: dict) -> List[str]:
        pass

    def generate(self, query: str, metadata: dict, retrieved_contexts: List[str]) -> str:
        pass
