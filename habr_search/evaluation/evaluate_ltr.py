"""проверяем сохранённый lightgbm на другом запуске без обучения и вызовов mws."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from habr_search.evaluation.comparison_data import load_context, load_judgments, load_records
from habr_search.evaluation.ltr_data import candidate_rows
from habr_search.evaluation.ranking_metrics import RankedDocument
from habr_search.evaluation.train_ltr import evaluate_predictions
from habr_search.retrieval.ltr import LearnedRanker
from habr_search.retrieval.tantivy_index import DEFAULT_INDEX, TantivySearch


class CliArgs(argparse.Namespace):
    from_run: Path
    qrels: Path
    model: Path
    output: Path
    tantivy_path: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', type=Path, required=True)
    parser.add_argument('--qrels', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tantivy-path', type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args(namespace=CliArgs())
    try:
        if args.output.exists():
            raise ValueError('папка отчёта уже существует')
        context = load_context(args.from_run)
        records = load_records(args.from_run, context)
        model = LearnedRanker(args.model)
        model.validate_context(context.collection, context.qdrant_revision, context.tantivy_revision, model.candidate_limit)
        if model.metadata.get('source_context_sha256') == hashlib.sha256((args.from_run / 'context.json').read_bytes()).hexdigest():
            raise ValueError('это обучающий запуск; для него смотрите validation с разделением по темам')
        snapshot = TantivySearch(args.tantivy_path)
        if snapshot.info.revision != context.tantivy_revision:
            raise ValueError('версия tantivy отличается от запуска')
        labels = load_judgments(context, args.qrels, include_seeds=False)
        start = perf_counter()
        catalog = snapshot.feature_catalog()
        rows = candidate_rows(context, records, catalog, model.candidate_limit)
        catalog_and_features_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        predictions = {query: tuple(RankedDocument(item.article_id, item.score) for item in model.predict(values))
                       for query, values in rows.items()}
        prediction_ms = (perf_counter() - start) * 1000
        evaluate_predictions(context, records, predictions, labels, args.output, model.candidate_limit)
        metadata = {
            'model_sha256': model.model_hash, 'source_run': str(args.from_run.resolve()),
            'rankings_sha256': hashlib.sha256((args.from_run / 'rankings.csv').read_bytes()).hexdigest(),
            'qrels_sha256': hashlib.sha256(args.qrels.read_bytes()).hexdigest(),
            'timing_scope': 'offline_catalog_features_and_predictions_not_http_latency',
            'catalog_and_features_ms': catalog_and_features_ms, 'prediction_ms': prediction_ms,
            'candidate_limit': model.candidate_limit, 'mws_calls': 0,
            'unjudged_policy': 'assumed_nonrelevant', 'label_status': 'preliminary_or_partial',
        }
        with (args.output / 'metadata.json').open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(f'отчёт: {args.output}; обращений к mws: 0; скорость http не измерялась')
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f'проверка ltr не завершена: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
