from sentence_transformers import CrossEncoder
from typing import List

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Инициализация реранкера.

        :param model_name: Имя модели CrossEncoder.
        """
        self.model_name = model_name
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query: str, results: list) -> str:
        """
        Реранкинг списка кандидатов.

        :param query: Запрос пользователя.
        :param results: Список результатов из Qdrant (list of dict),
                        где каждый элемент должен содержать ключ 'content'.
        :return: str
        """
        merge_best = ''
        for i in range(len(results)):
            candidates = [
                (query[i], result)  # Преобразуем в строку
                for result in results[i]
            ]

            # Инициализация реранкера
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # Пример модели для реранка

            # Оценка релевантности с помощью реранкера
            scores = reranker.predict(candidates)

            # Объединение результатов с оценками
            scored_results = [
                {"content": result, "score": score}
                for result, score in zip(results[i], scores)
            ]

            # Сортировка по убыванию релевантности
            scored_results = sorted(scored_results, key=lambda x: x['score'], reverse=True)

            merge_best += scored_results[0]['content'] + '\n-----------------------------------------------\n'

        return merge_best