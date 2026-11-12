"""считаем метрики ранжирования отдельно от индексов, модели и сетевых запросов."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RankedDocument:
    """одна статья занимает одно место независимо от числа фрагментов."""
    article_id: int
    score: float


@dataclass(frozen=True)
class PRPoint:
    """точка кривой после включения всех документов с данным score."""
    threshold: float | None  # none обозначает начало кривой без выбранных документов
    precision: float
    recall: float
    retrieved: int
    relevant: int


@dataclass(frozen=True)
class QueryMetrics:
    """храним метрики запроса и признаки неполноты ручной разметки."""
    values: dict[str, float]
    curve: tuple[PRPoint, ...]
    relevant_total: int
    returned: int
    unjudged_returned: int


def validate_ranking(ranking: Sequence[RankedDocument], universe: frozenset[int]) -> None:
    """не позволяем повторным id и повреждённым оценкам улучшить метрики."""
    seen: set[int] = set()
    previous = math.inf
    for item in ranking:
        if type(item.article_id) is not int or item.article_id not in universe or item.article_id in seen:
            raise ValueError('выдача содержит неизвестную или повторную статью')
        if not math.isfinite(item.score) or item.score > previous:
            raise ValueError('оценки должны быть конечными и идти по убыванию')
        seen.add(item.article_id)
        previous = item.score


def precision_recall_curve(
    ranking: Sequence[RankedDocument], relevant: frozenset[int], universe: frozenset[int],
    missing_score: float | None,
) -> tuple[PRPoint, ...]:
    """одинаковые оценки проходят порог вместе; отсутствующие статьи получают явную оценку."""
    # полноту проверяем по множеству документов, а не по глубине списка
    present = {item.article_id for item in ranking}
    missing = universe - present
    if missing and (missing_score is None or not math.isfinite(missing_score)):
        raise ValueError('для pr-auc нужны оценки всех статей или явная оценка отсутствующих')
    groups: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    for item in ranking:
        groups[item.score][0] += 1
        groups[item.score][1] += int(item.article_id in relevant)
    if missing and missing_score is not None:
        if ranking and missing_score > ranking[-1].score:
            raise ValueError('оценка отсутствующих не должна превышать оценку найденных')
        groups[missing_score][0] += len(missing)
        groups[missing_score][1] += len(missing & relevant)
    points = [PRPoint(None, 1.0, 0.0, 0, 0)]
    found = selected = 0
    for score in sorted(groups, reverse=True):
        count, hits = groups[score]
        selected += count
        found += hits
        points.append(PRPoint(score, found / selected, found / len(relevant), selected, found))
    return tuple(points)


def evaluate_ranking(
    ranking: Sequence[RankedDocument], judgments: Mapping[int, int], universe: frozenset[int],
    top_k: int = 5, cutoffs: Sequence[int] = (1, 3, 5, 10, 20), *, missing_score: float | None = None,
) -> QueryMetrics:
    """неразмеченные документы считаем нерелевантными, сохраняя число таких результатов."""
    if (not universe or any(type(value) is not int or value < 1 for value in universe)
            or type(top_k) is not int or top_k < 1 or not cutoffs
            or any(type(k) is not int or k < 1 for k in cutoffs) or len(set(cutoffs)) != len(cutoffs)):
        raise ValueError('нужны непустой корпус и разные положительные значения k')
    if any(type(i) is not int or i not in universe or type(g) is not int or not 0 <= g <= 3
           for i, g in judgments.items()):
        raise ValueError('разметка должна содержать id корпуса и целые оценки от 0 до 3')
    validate_ranking(ranking, universe)
    relevant = frozenset(i for i, grade in judgments.items() if grade > 0)
    selected = ranking[:top_k]
    unjudged = sum(item.article_id not in judgments for item in selected)
    if not relevant:
        # отсутствие ответа не смешиваем со средними метриками положительных запросов
        return QueryMetrics({}, (), 0, len(selected), unjudged)
    hits = 0
    hit_counts = [0]
    ap_sums = [0.0]
    dcg_sums = [0.0]
    first: int | None = None
    for rank, item in enumerate(ranking, 1):
        grade = judgments.get(item.article_id, 0)
        if grade > 0:
            hits += 1
            first = rank if first is None else first
        # вариант ndcg с экспоненциальным gain различает слабую и высокую релевантность
        hit_counts.append(hits)
        ap_sums.append(ap_sums[-1] + (hits / rank if grade > 0 else 0.0))
        dcg_sums.append(dcg_sums[-1] + (2 ** grade - 1) / math.log2(rank + 1))
    returned = len(selected)
    selected_hits = hit_counts[returned]
    precision = selected_hits / returned if returned else 0.0
    recall = selected_hits / len(relevant)
    values: dict[str, float] = {
        'RR': 1 / first if first is not None else 0.0,
        'Precision': precision, 'Recall': recall,
        'F1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        'AP': ap_sums[-1] / len(relevant),
    }
    ideal = sorted((2 ** grade - 1 for grade in judgments.values() if grade > 0), reverse=True)
    for k in cutoffs:
        depth = min(k, len(ranking))
        count = hit_counts[depth]
        p, r = count / k, count / len(relevant)
        values[f'Precision@{k}'] = p
        values[f'Recall@{k}'] = r
        values[f'F1@{k}'] = 2 * p * r / (p + r) if p + r else 0.0
        values[f'RR@{k}'] = 1 / first if first is not None and first <= k else 0.0
        # как в trec_eval map_cut: знаменатель — все известные релевантные статьи
        values[f'AP@{k}'] = ap_sums[depth] / len(relevant)
        idcg = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal[:k], 1))
        values[f'nDCG@{k}'] = dcg_sums[depth] / idcg
        values[f'Hit@{k}'] = float(count > 0)
    curve = precision_recall_curve(ranking, relevant, universe, missing_score)
    # pr-auc интегрируем трапециями; ap выше рассчитан отдельно и не подменяет площадь
    values['PR-AUC'] = math.fsum((right.recall - left.recall) * (left.precision + right.precision) / 2
                                for left, right in zip(curve, curve[1:]))
    return QueryMetrics(values, curve, len(relevant), returned, unjudged)


def aggregate_name(name: str) -> str:
    """обратное место и ap становятся mrr и map только после усреднения по запросам."""
    if name == 'RR' or name.startswith('RR@'):
        return 'MRR' + name[2:]
    if name == 'AP' or name.startswith('AP@'):
        return 'mAP' + name[2:]
    return name
