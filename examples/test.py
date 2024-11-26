import sys

from lear.custom_llm import CustomLlm
from lear.http_rag_provider import HttpRagProvider

sys.path.append('../lear')
from lear.evaluator import Evaluator


def measure():
    mode = 'rag'
    rag_provider = HttpRagProvider(mode)
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
    evaluate()
