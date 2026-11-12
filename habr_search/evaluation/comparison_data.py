"""сохраняем полные ранжирования и читаем разметку без подключения к поисковым сервисам."""
from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from habr_search.evaluation.evaluate import EvaluationQuery, query_set_hash
from habr_search.evaluation.ranking_metrics import RankedDocument, validate_ranking
from habr_search.retrieval.hybrid import HybridConfig, fuse_candidates

ENGINES: tuple[str, ...] = ('bge-m3', 'tantivy', 'hybrid')
COLUMNS: tuple[str, ...] = ('query_id', 'started_at', 'dense_ids', 'dense_scores', 'lexical_ids', 'lexical_scores',
                           'hybrid_ids', 'hybrid_scores', 'dense_ms', 'lexical_ms', 'hybrid_ms')


@dataclass(frozen=True)
class ComparisonContext:
    """фиксируем одинаковые запросы, статьи и настройки для всех трёх режимов."""
    collection: str
    qdrant_revision: str
    tantivy_revision: str
    points: int
    universe: frozenset[int]
    cases: tuple[EvaluationQuery, ...]
    config: HybridConfig

    def save(self, folder: Path) -> None:
        """метаданные содержат только настройки и id; тексты корпуса и ключ в них не копируются."""
        folder.mkdir(parents=True, exist_ok=False)
        data: dict[str, object] = {
            'format_version': 1, 'collection': self.collection, 'qdrant_revision': self.qdrant_revision,
            'tantivy_revision': self.tantivy_revision, 'points': self.points, 'article_ids': sorted(self.universe),
            'query_set_sha256': query_set_hash(self.cases),
            'queries': [{'query_id': q.query_id, 'category': q.category, 'query': q.query,
                         'relevant_article_ids': list(q.relevant_ids)} for q in self.cases],
            'candidate_limit': self.config.candidate_limit, 'rrf_k': self.config.rrf_k,
            'dense_weight': self.config.dense_weight, 'lexical_weight': self.config.lexical_weight,
            'ranking_scope': 'full_article_rankings',
            'missing_scores': {'bge-m3': None, 'tantivy': 0, 'hybrid': 0},
            'timing_scope': 'full_ranking_without_snippets_not_web_latency',
        }
        with (folder / 'context.json').open('x', encoding='utf-8') as target:
            json.dump(data, target, ensure_ascii=False, indent=2)
            target.write('\n')
        with (folder / 'rankings.csv').open('x', encoding='utf-8', newline='') as target:
            csv.writer(target).writerow(COLUMNS)


def json_mapping(value: object) -> dict[str, object]:
    """проверяем структуру json до обращения к полям метаданных."""
    if not isinstance(value, dict):
        raise ValueError('повреждены метаданные сравнения: ожидался объект')
    mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError('ключи метаданных должны быть строками')
    return cast(dict[str, object], value)


def json_list(value: object) -> list[object]:
    """сужаем неизвестное значение до проверенного списка."""
    if not isinstance(value, list):
        raise ValueError('повреждены метаданные сравнения: ожидался список')
    return cast(list[object], value)


def positive_ids(value: object) -> tuple[int, ...]:
    """отбрасываем неверные id, логические значения и повторы."""
    values = json_list(value)
    if any(type(i) is not int or i < 1 for i in values) or len(set(cast(list[int], values))) != len(values):
        raise ValueError('нужны разные положительные id статей')
    return tuple(cast(list[int], values))


def required_string(data: Mapping[str, object], key: str) -> str:
    """обязательное текстовое поле не должно быть пустым."""
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f'в метаданных отсутствует строка {key}')
    return value


def number(data: Mapping[str, object], key: str) -> float:
    """числа настроек должны быть конечными и не подменяться логическим типом."""
    value = data.get(key)
    if type(value) not in (int, float) or not math.isfinite(cast(float, value)):
        raise ValueError(f'в метаданных повреждено число {key}')
    return float(cast(float, value))


def load_context(folder: Path) -> ComparisonContext:
    """проверяем версию, запросы и состав корпуса перед чтением сохранённых результатов."""
    with (folder / 'context.json').open(encoding='utf-8') as source:
        data = json_mapping(json.load(source))
    if type(data.get('format_version')) is not int or data['format_version'] != 1:
        raise ValueError('неподдерживаемая версия сравнения')
    if (data.get('ranking_scope') != 'full_article_rankings'
            or data.get('missing_scores') != {'bge-m3': None, 'tantivy': 0, 'hybrid': 0}
            or data.get('timing_scope') != 'full_ranking_without_snippets_not_web_latency'):
        raise ValueError('изменены правила ранжирования или измерения времени')
    universe = frozenset(positive_ids(data.get('article_ids')))
    points, limit = data.get('points'), data.get('candidate_limit')
    if not universe or type(points) is not int or points < len(universe) or type(limit) is not int:
        raise ValueError('повреждены размеры корпуса или число кандидатов')
    cases: list[EvaluationQuery] = []
    for raw in json_list(data.get('queries')):
        item = json_mapping(raw)
        case = EvaluationQuery(required_string(item, 'query_id'), required_string(item, 'category'),
                               required_string(item, 'query'), positive_ids(item.get('relevant_article_ids')))
        if (bool(case.relevant_ids) == (case.category == 'no_answer')
                or not set(case.relevant_ids) <= universe):
            raise ValueError('разметка запроса противоречит категории или корпусу')
        cases.append(case)
    if (not cases or len({q.query_id for q in cases}) != len(cases)
            or len({q.query for q in cases}) != len(cases) or data.get('query_set_sha256') != query_set_hash(cases)):
        raise ValueError('повреждён набор запросов сравнения')
    return ComparisonContext(required_string(data, 'collection'), required_string(data, 'qdrant_revision'),
                             required_string(data, 'tantivy_revision'), points, universe, tuple(cases),
                             HybridConfig(limit, number(data, 'rrf_k'), number(data, 'dense_weight'), number(data, 'lexical_weight')))


@dataclass(frozen=True)
class ComparisonRecord:
    """храним исходные оценки и измеренное время одного завершённого запроса."""
    query_id: str
    started_at: str
    dense: tuple[RankedDocument, ...]
    lexical: tuple[RankedDocument, ...]
    hybrid: tuple[RankedDocument, ...]
    dense_ms: float
    lexical_ms: float
    hybrid_ms: float

    def ranking(self, engine: str) -> tuple[RankedDocument, ...]:
        """выбираем сохранённый порядок нужного режима."""
        return {'bge-m3': self.dense, 'tantivy': self.lexical, 'hybrid': self.hybrid}[engine]

    def elapsed(self, engine: str) -> float:
        """возвращаем длительность соответствующего полного ранжирования."""
        return {'bge-m3': self.dense_ms, 'tantivy': self.lexical_ms, 'hybrid': self.hybrid_ms}[engine]

    def row(self) -> list[str]:
        """сохраняем оценки без округления, чтобы не создавать искусственные ничьи."""
        values = [self.query_id, self.started_at]
        for ranking in (self.dense, self.lexical, self.hybrid):
            values += [';'.join(str(d.article_id) for d in ranking), ';'.join(repr(d.score) for d in ranking)]
        return values + [repr(self.dense_ms), repr(self.lexical_ms), repr(self.hybrid_ms)]


def parse_ranking(ids: str, scores: str, universe: frozenset[int]) -> tuple[RankedDocument, ...]:
    """связываем id и оценки и проверяем полный порядок строки csv."""
    identifiers = tuple(map(int, ids.split(';'))) if ids else ()
    values = tuple(map(float, scores.split(';'))) if scores else ()
    if len(identifiers) != len(values):
        raise ValueError('число оценок отличается от числа статей')
    ranking = tuple(RankedDocument(i, score) for i, score in zip(identifiers, values))
    validate_ranking(ranking, universe)
    return ranking


def load_records(folder: Path, context: ComparisonContext) -> dict[str, ComparisonRecord]:
    """незавершённый запуск можно продолжить; повреждённые строки повторно не оплачиваем."""
    records: dict[str, ComparisonRecord] = {}
    known = {q.query_id for q in context.cases}
    with (folder / 'rankings.csv').open(encoding='utf-8-sig', newline='') as source:
        reader = csv.reader(source, strict=True)
        if tuple(next(reader, list[str]())) != COLUMNS:
            raise ValueError('неверные столбцы сохранённых ранжирований')
        for row in reader:
            if len(row) != len(COLUMNS):
                raise ValueError('повреждена строка ранжирования')
            key, stamp = row[:2]
            if key not in known or key in records:
                raise ValueError('неизвестный или повторный запрос в отчёте')
            datetime.fromisoformat(stamp)
            dense, lexical, hybrid = (parse_ranking(row[i], row[i + 1], context.universe) for i in (2, 4, 6))
            if frozenset(d.article_id for d in dense) != context.universe or any(d.score < 0 for d in lexical):
                raise ValueError('нужна полная векторная выдача и неотрицательные оценки tantivy')
            fused = fuse_candidates([d.article_id for d in dense[:context.config.candidate_limit]],
                                    [d.article_id for d in lexical[:context.config.candidate_limit]], context.config)
            if hybrid != tuple(RankedDocument(d.article_id, d.score) for d in fused):
                raise ValueError('гибридная выдача не соответствует кандидатам и весам')
            dense_ms, lexical_ms, hybrid_ms = map(float, row[8:])
            if any(not math.isfinite(t) or t < 0 for t in (dense_ms, lexical_ms, hybrid_ms)):
                raise ValueError('повреждено время запроса')
            records[key] = ComparisonRecord(key, stamp, dense, lexical, hybrid, dense_ms, lexical_ms, hybrid_ms)
    return records


def load_judgments(context: ComparisonContext, path: Path | None) -> dict[str, dict[int, int]]:
    """расширяем начальную разметку; явный ноль позволяет снять ошибочную исходную метку."""
    judgments = {q.query_id: dict.fromkeys(q.relevant_ids, 1) for q in context.cases}
    if path is None:
        return judgments
    seen: set[tuple[str, int]] = set()
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        if not {'query_id', 'article_id', 'relevance'} <= set(reader.fieldnames or []):
            raise ValueError('разметке нужны столбцы query_id, article_id, relevance')
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('повреждена строка разметки')
            key, identifier, grade = row['query_id'], int(row['article_id']), row['relevance'].strip()
            if key not in judgments or identifier not in context.universe or (key, identifier) in seen:
                raise ValueError('разметка содержит неизвестный запрос, статью или повтор')
            seen.add((key, identifier))
            if grade:  # пустая ячейка остаётся неразмеченной
                value = int(grade)
                if not 0 <= value <= 3:
                    raise ValueError('relevance должна быть от 0 до 3')
                judgments[key][identifier] = value
    for case in context.cases:
        has_positive = any(grade > 0 for grade in judgments[case.query_id].values())
        if has_positive == (case.category == 'no_answer'):
            raise ValueError(f'{case.query_id}: новые метки противоречат категории; уточните список ответов или набор запросов')
    return judgments


def export_pool(context: ComparisonContext, records: Mapping[str, ComparisonRecord],
                judgments: Mapping[str, Mapping[int, int]], output: Path, depth: int) -> None:
    """объединяем кандидатов всех режимов, оставляя неизвестные оценки пустыми."""
    with output.open('x', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['query_id', 'query', 'article_id', 'url', 'bge_m3_rank', 'tantivy_rank',
                         'hybrid_rank', 'relevance', 'comment'])
        for case in context.cases:
            record = records[case.query_id]
            ranks = [{d.article_id: rank for rank, d in enumerate(record.ranking(engine), 1)} for engine in ENGINES]
            pool = set(judgments[case.query_id])
            for engine in ENGINES:
                pool.update(d.article_id for d in record.ranking(engine)[:depth])
            for identifier in sorted(pool):
                writer.writerow([case.query_id, case.query, identifier, f'https://habr.com/ru/articles/{identifier}/',
                                 *(rank.get(identifier, '') for rank in ranks), judgments[case.query_id].get(identifier, ''), ''])
