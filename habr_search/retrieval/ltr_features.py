"""одинаково вычисляем признаки кандидатов при обучении и на сервере."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from habr_search.retrieval.router import STOP_WORDS, WORDS, normalize, query_identifiers

FEATURE_VERSION: str = 'ltr-features-v1'
FEATURE_NAMES: tuple[str, ...] = (
    'dense_score', 'dense_reciprocal_rank', 'dense_present',
    'lexical_log_score', 'lexical_relative_score', 'lexical_reciprocal_rank', 'lexical_present', 'both_sources',
    'title_coverage', 'body_coverage', 'title_idf_coverage', 'body_idf_coverage',
    'title_identifier_coverage', 'body_identifier_coverage', 'query_terms_log', 'body_length_log',
)


@dataclass(frozen=True)
class RankSignal:
    article_id: int
    score: float


@dataclass(frozen=True)
class FeatureDocument:
    title: str
    title_terms: frozenset[str]
    body: str
    body_terms: frozenset[str]

    @classmethod
    def from_text(cls, title: str, body: str) -> FeatureDocument:
        """готовим текстовые признаки один раз при открытии неизменяемого индекса."""
        title, body = normalize(title), normalize(body)
        return cls(title, frozenset(WORDS.findall(title)), body, frozenset(WORDS.findall(body)))


class FeatureCatalog:
    """каталог принадлежит снимку tantivy; при поиске корпус повторно не читается."""
    def __init__(self, documents: Mapping[int, FeatureDocument]) -> None:
        if not documents or any(type(identifier) is not int or identifier < 1 for identifier in documents):
            raise ValueError('для признаков нужен непустой каталог статей с положительными id')
        self.documents: dict[int, FeatureDocument] = dict(documents)
        frequencies: Counter[str] = Counter()
        for document in documents.values():
            frequencies.update(document.body_terms | document.title_terms)
        self.idf: dict[str, float] = {term: math.log1p(len(documents) / count) for term, count in frequencies.items()}
        self.unseen_idf: float = math.log1p(len(documents))


@dataclass(frozen=True)
class CandidateFeatures:
    article_id: int  # id используется для связи с выдачей, но не входит в матрицу модели
    values: tuple[float, ...]
    dense_rank: int | None
    lexical_rank: int | None


def literal_match(text: str, needle: str) -> bool:
    """ищем строкой и проверяем границы слов; не сканируем каждый символ регулярным выражением."""
    if not needle:
        return False
    position = text.find(needle)
    while position >= 0:
        end = position + len(needle)
        left = position == 0 or not (text[position - 1].isalnum() or text[position - 1] == '_')
        right = end == len(text) or not (text[end].isalnum() or text[end] == '_')
        if left and right:
            return True
        position = text.find(needle, position + 1)
    return False


def build_features(query: str, dense: Sequence[RankSignal], lexical: Sequence[RankSignal],
                   catalog: FeatureCatalog, limit: int) -> list[CandidateFeatures]:
    """отсутствующую оценку кодируем нулём и отдельным флагом наличия источника."""
    if not 1 <= limit <= 100:
        raise ValueError('глубина кандидатов должна быть от 1 до 100')
    for source in (dense, lexical):
        ids = [item.article_id for item in source[:limit]]
        if len(ids) != len(set(ids)) or any(identifier not in catalog.documents for identifier in ids):
            raise ValueError('кандидаты содержат повтор или неизвестную статью')
        if any(not math.isfinite(item.score) for item in source[:limit]):
            raise ValueError('оценки кандидатов должны быть конечными')
    d = {item.article_id: (rank, item.score) for rank, item in enumerate(dense[:limit], 1)}
    l = {item.article_id: (rank, item.score) for rank, item in enumerate(lexical[:limit], 1)}
    if any(score < 0 for _, score in l.values()):
        raise ValueError('bm25 не должен быть отрицательным')
    terms = set(WORDS.findall(normalize(query))) - STOP_WORDS
    weights = {term: catalog.idf.get(term, catalog.unseen_idf) for term in terms}
    total_idf = sum(weights.values())
    identifiers = query_identifiers(query)
    maximum = max((score for _, score in l.values()), default=0.0)
    result: list[CandidateFeatures] = []
    for identifier in sorted(d.keys() | l.keys()):
        document = catalog.documents[identifier]
        dense_rank, dense_score = d.get(identifier, (0, 0.0))
        lexical_rank, lexical_score = l.get(identifier, (0, 0.0))
        title_hits, body_hits = terms & document.title_terms, terms & document.body_terms
        values = (
            dense_score, 1 / dense_rank if dense_rank else 0.0, float(bool(dense_rank)),
            math.log1p(lexical_score), lexical_score / maximum if maximum else 0.0,
            1 / lexical_rank if lexical_rank else 0.0, float(bool(lexical_rank)),
            float(bool(dense_rank and lexical_rank)),
            len(title_hits) / len(terms) if terms else 0.0, len(body_hits) / len(terms) if terms else 0.0,
            sum(weights[term] for term in title_hits) / total_idf if total_idf else 0.0,
            sum(weights[term] for term in body_hits) / total_idf if total_idf else 0.0,
            sum(literal_match(document.title, item) for item in identifiers) / len(identifiers) if identifiers else 0.0,
            sum(literal_match(document.body, item) for item in identifiers) / len(identifiers) if identifiers else 0.0,
            math.log1p(len(terms)), math.log1p(len(document.body)),
        )
        result.append(CandidateFeatures(identifier, values, dense_rank or None, lexical_rank or None))
    return result
