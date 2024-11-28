import os
import sys

from lear.custom_llm import CustomLlm
from providers.http_rag_provider import HttpRagProvider

sys.path.append('../lear')
from lear.evaluator import Evaluator


def measure():
    mode = 'rag'
    url = 'http://305-server:5566'
    rag_provider = HttpRagProvider(mode, url, access_token='abc123')
    llm = CustomLlm()
    evaluator = Evaluator(llm)
    source_file_path = 'dataset/Corpus-L1.json'

    measure_result = evaluator.measure(source_file_path, rag_provider)
    print(measure_result)


def evaluate():
    llm = CustomLlm()
    evaluator = Evaluator(llm)

    check_point_file_path = '/tmp/rag_Corpus-L1_measure.json'
    result = evaluator.eval(check_point_file_path)
    print(result)


if __name__ == '__main__':
    api_key = "9e3d5d7d3ff2e74fc9c7e0684c3b6555.feSMhxTcgznEB07f"
    base_url = "https://open.bigmodel.cn/api/paas/v4/"
    os.environ['OPENAI_API_PROVIDER'] = 'glm-4-air'
    os.environ['OPENAI_API_KEY'] = api_key
    os.environ['OPENAI_API_BASE'] = base_url
    evaluate()
