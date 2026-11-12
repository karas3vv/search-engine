"""создаём таблицы метрик, pr-кривых и кандидатов для расширения разметки."""
from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

from habr_search.evaluation.comparison_data import (
    ENGINES, ComparisonContext, ComparisonRecord, export_pool, load_judgments,
)
from habr_search.evaluation.ranking_metrics import QueryMetrics, aggregate_name, evaluate_ranking


def create_metrics(
    context: ComparisonContext, records: Mapping[str, ComparisonRecord], destination: Path,
    top_k: int, cutoffs: Sequence[int], qrels: Path | None = None, pool_depth: int = 20,
) -> None:
    """пересчитываем сохранённый запуск; исходные ранжирования и разметку не изменяем."""
    if destination.exists():
        raise ValueError('папка метрик уже существует; укажите новый --report-output')
    if not 1 <= top_k <= 20 or not cutoffs or any(not 1 <= k <= 20 for k in cutoffs) or pool_depth < 1:
        raise ValueError('top-k и значения k должны быть от 1 до 20; pool-depth должен быть положительным')
    if len(set(cutoffs)) != len(cutoffs) or set(records) != {case.query_id for case in context.cases}:
        raise ValueError('нужны разные значения k и завершённая оценка всех запросов')
    judgments = load_judgments(context, qrels)
    scored: dict[tuple[str, str], QueryMetrics] = {}
    # все варианты используют одну разметку; оценки разных движков не смешиваются
    for engine in ENGINES:
        for case in context.cases:
            scored[engine, case.query_id] = evaluate_ranking(
                records[case.query_id].ranking(engine), judgments[case.query_id], context.universe,
                top_k, cutoffs, missing_score=None if engine == 'bge-m3' else 0.0,
            )
    metric_names = list(next(value.values for value in scored.values() if value.values)) if any(v.values for v in scored.values()) else []
    destination.mkdir(parents=True)
    provenance = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'top_k_for_set_metrics': str(top_k),
        'cutoffs': ';'.join(map(str, cutoffs)), 'articles': str(len(context.universe)),
        'queries': str(len(context.cases)), 'collection': context.collection, 'qdrant_revision': context.qdrant_revision,
        'tantivy_revision': context.tantivy_revision,
        'label_status': 'complete' if all(frozenset(g) == context.universe for g in judgments.values()) else 'partial_seed_or_pooled',
        'unjudged_policy': 'assumed_nonrelevant', 'positive_threshold': 'relevance > 0',
        'ndcg_gain': '2**relevance - 1', 'ap_at_k_denominator': 'all_known_relevant_documents',
        'pr_auc_method': 'per_query_trapezoidal_score_thresholds_with_ties',
        'unretrieved_scores': 'tantivy=0;hybrid=0;bge-m3=complete_scores',
        'quality_averaging': 'macro_over_queries_with_positive_labels;negative_queries_separate',
        'timing_scope': 'full_ranking_without_snippets_not_web_latency',
        'qrels_sha256': hashlib.sha256(qrels.read_bytes()).hexdigest() if qrels else 'seed_labels_from_queries',
        'corpus_label_coverage': str(sum(len(g) for g in judgments.values()) / (len(context.universe) * len(context.cases))),
    }
    with (destination / 'metadata.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['setting', 'value'])
        writer.writerows(provenance.items())
    with (destination / 'per_query.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['engine', 'query_id', 'category', 'query', 'relevant_total', 'returned_at_top_k',
                         'unjudged_at_top_k', 'elapsed_full_ranking_ms', *metric_names])
        for engine in ENGINES:
            for case in context.cases:
                score = scored[engine, case.query_id]
                writer.writerow([engine, case.query_id, case.category, case.query, score.relevant_total,
                                 score.returned, score.unjudged_returned, records[case.query_id].elapsed(engine),
                                 *(score.values.get(key, '') for key in metric_names)])
    with (destination / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['engine', 'category', 'queries', 'positive_queries', 'negative_queries',
                         'negative_empty', 'median_full_ranking_ms', 'p95_full_ranking_ms',
                         *(aggregate_name(key) for key in metric_names)])
        for engine in ENGINES:
            for category in ('all', *sorted({case.category for case in context.cases})):
                cases = [case for case in context.cases if category == 'all' or case.category == category]
                values = [scored[engine, case.query_id] for case in cases]
                positives = [value for value in values if value.relevant_total]
                negatives = [value for value in values if not value.relevant_total]
                times = sorted(records[case.query_id].elapsed(engine) for case in cases)
                averages = [math.fsum(value.values[key] for value in positives) / len(positives) if positives else ''
                            for key in metric_names]
                writer.writerow([engine, category, len(cases), len(positives), len(negatives),
                                 sum(not value.returned for value in negatives), median(times),
                                 times[max(0, math.ceil(len(times) * .95) - 1)], *averages])
                if category == 'all':
                    print(f'{engine}: положительных запросов {len(positives)}; отрицательных {len(negatives)}')
                    if negatives:
                        print(f'  no_answer: пустая выдача {sum(not value.returned for value in negatives)}/{len(negatives)}')
                    for name, average in zip(metric_names, averages):
                        if isinstance(average, float):
                            print(f'  {aggregate_name(name)}: {average:.4f}')
                    print(f'  время полного ранжирования: медиана {median(times):.0f} мс; '
                          f'p95 {times[max(0, math.ceil(len(times) * .95) - 1)]:.0f} мс')
    with (destination / 'pr_curves.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['engine', 'query_id', 'threshold', 'precision', 'recall', 'retrieved', 'relevant'])
        for (engine, query_id), score in scored.items():
            for point in score.curve:
                writer.writerow([engine, query_id, '' if point.threshold is None else point.threshold,
                                 point.precision, point.recall, point.retrieved, point.relevant])
    export_pool(context, records, judgments, destination / 'judgments.csv', pool_depth)
    if provenance['label_status'] != 'complete':
        print('разметка неполная: это предварительные метрики; неразмеченные статьи считаются нерелевантными')
    print(f'метрики: {destination / "summary.csv"}')
    print(f'разметка нескольких статей на запрос: {destination / "judgments.csv"}')
