import os
import time
from pathlib import Path
from typing import List

import jsonpickle
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from lear import utils
from lear.metric import Metric
from lear.rag_provider import RAGProvider


class Evaluator(object):
    def __init__(self, llm):
        self.custom_llm = llm

    def measure(self, path: str, rag_provider: RAGProvider):
        files = []
        if os.path.isdir(path):
            sub_files = os.listdir(path)
            for file in sub_files:
                files.append(os.path.join(path, file))
        elif os.path.isfile(path):
            files.append(path)
        else:
            exit(1)
        result = {}
        for file in files:
            measure_result = self.__measure(file, rag_provider)
            file_name = Path(file).name
            check_point_file_path = f"/tmp/{rag_provider.name}_{Path(file).stem}_measure.json"
            self.__save(check_point_file_path, measure_result)
            result[file_name] = {'result': measure_result, 'check_point_file_path': check_point_file_path}
        return result

    def eval(self, file_path: str):
        result = {}
        negative_rejection_metrics = {'total_count': 0, 'successful_count': 0}
        misleading_metrics = {'total_count': 0, 'successful_count': 0}
        count = 0
        with open(file_path, 'r') as file:
            items = jsonpickle.decode(file.read())
        print(f"total items: {len(items)}")
        for chunk in items:
            q_type = chunk['type']
            metrics = chunk.get('metrics')
            for metric in metrics:
                name = metric.name
                if q_type == 'NegativeRejection':
                    negative_rejection_metrics["total_count"] += 1
                    if name == 'Correctness' and metric.score >= 0.5:
                        negative_rejection_metrics["successful_count"] += 1
                else:
                    if q_type == 'Irrelevant' or q_type == 'EvidentConflict':
                        misleading_metrics["total_count"] += 1
                        if name == 'Correctness' and metric.score >= 0.5:
                            misleading_metrics["successful_count"] += 1
                    if result.get(name) is None:
                        result[metric.name] = metric.score
                    else:
                        result[metric.name] = result[metric.name] + metric.score
            if q_type != 'NegativeRejection':
                count += 1

        print(count)
        result = {k: v / count for k, v in result.items()}
        if negative_rejection_metrics['total_count'] != 0:
            result['NegativeRejection'] = negative_rejection_metrics['successful_count'] / negative_rejection_metrics[
                'total_count']
        if misleading_metrics['total_count'] != 0:
            result['MisleadingRate'] = misleading_metrics['successful_count'] / misleading_metrics[
                'total_count']
        return result

    def __save(self, check_point_file_name, result):
        with open(check_point_file_name, "w") as outfile:
            outfile.write(jsonpickle.encode(result))

    def __measure(self, file_path: str, rag_provider: RAGProvider):
        chunks = []
        items = utils.read_file_content(file_path)
        for item in items:
            start = time.perf_counter()
            query = item.get("question")
            metadata = {}
            expected_contexts = item.get("positive_contexts")
            golden_answer = item.get("answer")
            q_type = item.get("type")
            print(f"current item:{q_type}:{query}")
            retrieved_contexts = rag_provider.retrieve(query, metadata)
            retrieval_latency = time.perf_counter() - start
            generated_answer = rag_provider.generate(query, metadata, retrieved_contexts)
            total_latency = time.perf_counter() - start
            # compute metrics for retrieval
            retrieval_metrics = self.compute_metrics_for_retrieval(expected_contexts, retrieved_contexts)
            if q_type != 'NegativeRejection':
                generation_metrics = self.compute_metrics_for_generation(query, golden_answer, generated_answer,
                                                                         expected_contexts, retrieved_contexts)
            else:
                generation_metrics = self.compute_metrics_for_negative_rejecttion(query, golden_answer,
                                                                                  generated_answer,
                                                                                  expected_contexts, retrieved_contexts)
            latency_metrics = [
                Metric('TotalLatency', total_latency, 'System'),
                Metric('RetrievalLatency', retrieval_latency, 'System'),
                Metric('GenerationLatency', total_latency - retrieval_latency, 'System'),
            ]
            chunk = [*retrieval_metrics, *generation_metrics, *latency_metrics]
            chunks.append({'question': query, 'type': q_type, 'metrics': chunk})
        return chunks

    def compute_metrics_for_retrieval(self, expected_contexts, retrieved_contexts) -> List[Metric]:
        result = []
        metrics = utils.calculate_metrics(retrieved_contexts, expected_contexts)
        for k, v in metrics.items():
            result.append(Metric(k, v, 'Retrieval'))
        return result

    def compute_metrics_for_generation(self, question: str, golden_answer, generated_answer, expected_contexts,
                                       retrieved_contexts) -> \
            List[Metric]:
        result = []

        test_case = LLMTestCase(
            input=question,
            actual_output=generated_answer,
            expected_output=''.join(golden_answer),
            retrieval_context=retrieved_contexts,
            context=expected_contexts
        )

        correctness_metric = GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            # NOTE: you can only provide either criteria or evaluation_steps, and not both
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            model=self.custom_llm,
        )
        correctness_metric.measure(test_case)

        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=self.custom_llm)
        answer_relevancy_metric.measure(test_case)

        faithfulness_metric = FaithfulnessMetric(
            model=self.custom_llm,
            include_reason=True
        )
        faithfulness_metric.measure(test_case)

        result.append(Metric("Correctness", correctness_metric.score, 'Generation'))
        result.append(Metric("AnswerRelevancy", answer_relevancy_metric.score, 'Generation'))
        result.append(Metric("Faithfulness", faithfulness_metric.score, 'Generation'))
        return result

    def compute_metrics_for_negative_rejecttion(self, question: str, golden_answer, generated_answer, expected_contexts,
                                                retrieved_contexts) -> \
            List[Metric]:
        result = []
        correctness_score = 0
        for answer in golden_answer:
            if answer in generated_answer:
                correctness_score = 1
        result.append(Metric("Correctness", correctness_score, 'Generation'))
        result.append(Metric("AnswerRelevancy", 0, 'Generation'))
        result.append(Metric("Faithfulness", 0, 'Generation'))
        return result
