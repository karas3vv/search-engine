"""обучаем lightgbm на явной разметке и проверяем целые темы по очереди без mws."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import cast

from habr_search.evaluation.comparison_data import ComparisonContext, ComparisonRecord, load_context, load_judgments, load_records
from habr_search.evaluation.ltr_data import candidate_rows, fit_ranker, load_topics, training_groups
from habr_search.evaluation.ranking_metrics import RankedDocument, aggregate_name, evaluate_ranking
from habr_search.paths import PROJECT_ROOT
from habr_search.retrieval.hybrid import HybridConfig, fuse_candidates
from habr_search.retrieval.ltr import BoosterAPI, predict_candidates, require_lightgbm
from habr_search.retrieval.ltr_features import FEATURE_NAMES, FEATURE_VERSION
from habr_search.retrieval.tantivy_index import DEFAULT_INDEX, TantivySearch


def evaluate_predictions(context: ComparisonContext, records: Mapping[str, ComparisonRecord],
                         predictions: Mapping[str, Sequence[RankedDocument]],
                         judgments: Mapping[str, Mapping[int, int]], output: Path, limit: int) -> None:
    """сравниваем одинаковые запросы; rrf пересчитываем с той же глубиной кандидатов."""
    output.mkdir(parents=True, exist_ok=False)
    values: list[tuple[str, str, dict[str, float]]] = []
    for case in context.cases:
        record = records[case.query_id]
        config = HybridConfig(limit, context.config.rrf_k, context.config.dense_weight, context.config.lexical_weight)
        fused = tuple(RankedDocument(item.article_id, item.score) for item in fuse_candidates(
            [item.article_id for item in record.dense[:limit]], [item.article_id for item in record.lexical[:limit]], config))
        for engine, ranking in (('bge-m3', record.dense), ('tantivy', record.lexical),
                                 ('hybrid', fused), ('ltr', predictions[case.query_id])):
            # отрицательные сырые оценки ltr допустимы; отсутствующие документы помещаются ниже последнего
            missing = min((item.score for item in ranking), default=0.0) - 1.0 if engine == 'ltr' else 0.0
            score = evaluate_ranking(ranking, judgments[case.query_id], context.universe, cutoffs=(5, 10),
                                     missing_score=None if engine == 'bge-m3' else missing)
            values.append((engine, case.query_id, score.values))
    metrics = list(next((score for _, _, score in values if score), dict[str, float]()))
    with (output / 'per_query.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['engine', 'query_id', *metrics])
        writer.writerows([engine, query, *(score.get(metric, '') for metric in metrics)] for engine, query, score in values)
    with (output / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['engine', 'queries_with_positives', *(aggregate_name(metric) for metric in metrics)])
        for engine in ('bge-m3', 'tantivy', 'hybrid', 'ltr'):
            scores = [score for current, _, score in values if current == engine and score]
            writer.writerow([engine, len(scores), *(mean(score[metric] for score in scores) if scores else '' for metric in metrics)])
            if scores:
                print(engine + ': ' + '; '.join(f'{aggregate_name(metric)}={mean(score[metric] for score in scores):.4f}'
                      for metric in ('RR', 'Precision@5', 'Recall@10', 'nDCG@10')))


class CliArgs(argparse.Namespace):
    from_run: Path
    qrels: Path
    groups: Path
    tantivy_path: Path
    output: Path
    candidate_limit: int
    trees: int


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', type=Path, required=True)
    parser.add_argument('--qrels', type=Path, required=True)
    parser.add_argument('--groups', type=Path, default=PROJECT_ROOT / 'evaluation/ltr_topics.csv')
    parser.add_argument('--tantivy-path', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-limit', type=int, default=50)
    parser.add_argument('--trees', type=int, default=80)
    args = parser.parse_args(namespace=CliArgs())
    try:
        if args.output.exists() or not 20 <= args.candidate_limit <= 100:
            raise ValueError('нужны новая папка output и candidate-limit от 20 до 100')
        require_lightgbm()
        import lightgbm as lgb
        context = load_context(args.from_run)
        records = load_records(args.from_run, context)
        labels = load_judgments(context, args.qrels, include_seeds=False)
        topics = load_topics(args.groups, {case.query_id for case in context.cases})
        snapshot = TantivySearch(args.tantivy_path)
        if snapshot.info.revision != context.tantivy_revision:
            raise ValueError('версия tantivy отличается от сохранённого запуска')
        start = perf_counter()
        catalog = snapshot.feature_catalog()
        rows = candidate_rows(context, records, catalog, args.candidate_limit)
        groups = training_groups(rows, labels, topics)
        used_topics = {group.topic for group in groups}
        if len(used_topics) < 3 or set(topics) != {group.query_id for group in groups}:
            raise ValueError('нужны минимум три темы и различающиеся явные оценки хотя бы двух кандидатов каждого запроса')
        args.output.mkdir(parents=True)
        # промежуточные модели каждого фолда не сохраняются как готовые серверные артефакты
        predictions: dict[str, tuple[RankedDocument, ...]] = {}
        folds: list[dict[str, object]] = []
        for held_topic in sorted(used_topics):
            train = [group for group in groups if group.topic != held_topic]
            held_ids = [query for query, topic in topics.items() if topic == held_topic]
            booster = fit_ranker(train, args.trees)
            predictions.update({query: tuple(RankedDocument(item.article_id, item.score)
                                for item in predict_candidates(booster, rows[query], args.candidate_limit))
                                for query in held_ids})
            folds.append({'held_topic': held_topic, 'train_query_ids': [group.query_id for group in train],
                          'validation_query_ids': held_ids})
        evaluate_predictions(context, records, predictions, labels, args.output / 'validation', args.candidate_limit)
        final = fit_ranker(groups, args.trees)
        final.save_model(str(args.output / 'model.txt'))
        metadata: dict[str, object] = {
            'format_version': 1, 'feature_version': FEATURE_VERSION, 'feature_names': list(FEATURE_NAMES),
            'objective': 'lambdarank', 'lightgbm_version': lgb.__version__,
            'candidate_limit': args.candidate_limit, 'collection': context.collection,
            'qdrant_revision': context.qdrant_revision, 'tantivy_revision': context.tantivy_revision,
            'model_sha256': hashlib.sha256((args.output / 'model.txt').read_bytes()).hexdigest(),
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'trees_requested': args.trees,
            'training_queries': len(groups), 'training_pairs': sum(len(group.rows) for group in groups),
            'unjudged_policy_training': 'excluded', 'validation_unjudged_policy': 'assumed_nonrelevant',
            'validation_strategy': 'leave_one_topic_out_fixed_parameters', 'folds': folds,
            'source_context_sha256': hashlib.sha256((args.from_run / 'context.json').read_bytes()).hexdigest(),
            'source_rankings_sha256': hashlib.sha256((args.from_run / 'rankings.csv').read_bytes()).hexdigest(),
            'qrels_sha256': hashlib.sha256(args.qrels.read_bytes()).hexdigest(),
            'groups_sha256': hashlib.sha256(args.groups.read_bytes()).hexdigest(),
            'experimental': True, 'label_status': 'preliminary_labels_require_review',
            'missing_features': 'score=0_and_presence_flag;dense_score_only_if_in_dense_candidates',
            'feature_importance_gain': dict(zip(FEATURE_NAMES,
                (float(value) for value in cast(BoosterAPI, final).feature_importance('gain')))),
            'elapsed_local_training_ms': (perf_counter() - start) * 1000, 'mws_calls': 0,
        }
        with (args.output / 'metadata.json').open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(f'модель: {args.output}; обучающих пар: {metadata["training_pairs"]}; запросов: {len(groups)}; тем: {len(used_topics)}')
        print('обращений к mws: 0; проверка предварительная; улучшение на новых запросах не гарантируется')
        return 0
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'обучение ltr не завершено: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
