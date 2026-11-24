"""переставляем первые статьи гибрида по оценке пары запрос — отрывок."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import requests

from habr_search.retrieval.hybrid import HybridResult

DEFAULT_RERANK_URL = 'https://gpt.mwsapis.ru/projects/habr-parser/cohere/v2/rerank'


class RerankError(RuntimeError):
    """запрос завершился ошибкой или модель вернула неполный порядок."""


def parse_order(payload: object, count: int) -> list[int]:
    """каждый отправленный документ должен встречаться в ответе ровно один раз."""
    rows = payload.get('results') if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != count:
        raise RerankError('неверное число результатов')
    scores: dict[int, float] = {}
    for row in cast(list[object], rows):
        if not isinstance(row, dict):
            raise RerankError('неверный элемент results')
        index, score = row.get('index'), row.get('relevance_score')
        if type(index) is not int or not 0 <= index < count or index in scores:
            raise RerankError('неверный индекс документа')
        if type(score) not in (int, float):
            raise RerankError('неверная оценка документа')
        try:
            value = float(score)
        except OverflowError:
            raise RerankError('слишком большая оценка документа') from None
        if not math.isfinite(value):
            raise RerankError('неверная оценка документа')
        scores[index] = value
    # при равной релевантности сохраняем исходный порядок rrf
    return sorted(scores, key=lambda index: (-scores[index], index))


class Reranker:
    def __init__(self, session: requests.Session, key: str, deployment: str, depth: int,
                 url: str = DEFAULT_RERANK_URL, timeout: float = 5.0) -> None:
        if not deployment.strip() or not 2 <= depth <= 20:
            raise ValueError('укажите имя деплоймента и глубину реранкинга от 2 до 20')
        self.session = session
        self.session.headers['Authorization'] = f'Bearer {key}'
        self.deployment = deployment
        self.depth = depth
        self.url = url
        self.timeout = timeout

    def rank(self, query: str, articles: Sequence[HybridResult]) -> list[HybridResult]:
        head = articles[:self.depth]
        if len(head) < 2:
            return list(articles)
        documents = [f'{item.title[:180]}\n{item.snippet[:500]}' for item in head]
        try:
            with self.session.post(self.url, json={'model': self.deployment, 'query': query,
                                   'documents': documents}, timeout=self.timeout, allow_redirects=False) as response:
                if response.status_code != 200:
                    raise RerankError(f'http {response.status_code}')
                order = parse_order(response.json(), len(head))
        except (requests.RequestException, ValueError) as exc:
            raise RerankError(type(exc).__name__) from None
        return [head[index] for index in order] + list(articles[self.depth:])
