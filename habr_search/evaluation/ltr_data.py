"""готовим размеченные группы ltr, не превращая неизвестные статьи в отрицательные."""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from habr_search.evaluation.comparison_data import ComparisonContext, ComparisonRecord
from habr_search.retrieval.ltr_features import FEATURE_NAMES, CandidateFeatures, FeatureCatalog, RankSignal, build_features

if TYPE_CHECKING:
    from lightgbm import Booster, Dataset


class Trainer(Protocol):
    """используем только проверяемую часть api, не распространяя unknown из callback-типов."""
    def train(self, params: dict[str, object], train_set: Dataset, *, num_boost_round: int) -> Booster: ...


@dataclass(frozen=True)
class LabeledGroup:
    query_id: str
    topic: str
    rows: tuple[CandidateFeatures, ...]
    grades: tuple[int, ...]


def load_topics(path: Path, query_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'query_id', 'topic'} <= set(reader.fieldnames or []):
            raise ValueError('groups нужны колонки query_id и topic')
        for row in reader:
            query, topic = row['query_id'], row['topic'].strip()
            if query not in query_ids or query in result or not topic:
                raise ValueError('groups содержит повтор, неизвестный запрос или пустую тему')
            result[query] = topic
    if set(result) != query_ids:
        raise ValueError('для каждого запроса нужна тема в groups')
    return result


def candidate_rows(context: ComparisonContext, records: Mapping[str, ComparisonRecord], catalog: FeatureCatalog,
                   limit: int) -> dict[str, list[CandidateFeatures]]:
    if frozenset(catalog.documents) != context.universe or set(records) != {case.query_id for case in context.cases}:
        raise ValueError('нужны тот же корпус и завершённые ранжирования всех запросов')
    return {case.query_id: build_features(case.query,
            [RankSignal(item.article_id, item.score) for item in records[case.query_id].dense[:limit]],
            [RankSignal(item.article_id, item.score) for item in records[case.query_id].lexical[:limit]], catalog, limit)
            for case in context.cases}


def training_groups(rows: Mapping[str, Sequence[CandidateFeatures]], grades: Mapping[str, Mapping[int, int]],
                    topics: Mapping[str, str]) -> list[LabeledGroup]:
    """loss получает только явные оценки 0–3; одинаковые оценки не создают обучающих пар."""
    result: list[LabeledGroup] = []
    for query_id, candidates in rows.items():
        selected = tuple(row for row in candidates if row.article_id in grades[query_id])
        labels = tuple(grades[query_id][row.article_id] for row in selected)
        if len(set(labels)) > 1:
            result.append(LabeledGroup(query_id, topics[query_id], selected, labels))
    return result


def fit_ranker(groups: Sequence[LabeledGroup], trees: int = 80) -> Booster:
    """параметры фиксированы до проверки; метки и строки каждого запроса идут одним блоком."""
    from habr_search.retrieval.ltr import require_lightgbm
    require_lightgbm()
    import lightgbm as lgb
    import numpy as np
    if len(groups) < 2 or trees < 1 or trees > 1000:
        raise ValueError('нужны хотя бы две обучаемые группы и от 1 до 1000 деревьев')
    matrix = np.asarray([row.values for group in groups for row in group.rows], dtype=np.float64)
    labels = np.asarray([grade for group in groups for grade in group.grades], dtype=np.int32)
    data = lgb.Dataset(matrix, label=labels, group=[len(group.rows) for group in groups],
                       feature_name=list(FEATURE_NAMES))
    params: dict[str, object] = {
        'objective': 'lambdarank', 'metric': 'ndcg', 'label_gain': [0, 1, 3, 7],
        'learning_rate': 0.05, 'num_leaves': 7, 'max_depth': 3, 'min_data_in_leaf': 5,
        'min_data_in_bin': 1, 'feature_pre_filter': False, 'lambdarank_truncation_level': 20,
        'seed': 17, 'num_threads': 1, 'deterministic': True, 'force_col_wise': True, 'verbosity': -1,
    }
    return cast(Trainer, lgb).train(params, data, num_boost_round=trees)
