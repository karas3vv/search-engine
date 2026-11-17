"""проверяем auto по сохранённой выдаче и заголовкам того же снимка tantivy без mws."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from habr_search.evaluation.comparison_data import (
    ENGINES, ComparisonContext, ComparisonRecord, load_context, load_judgments, load_records,
)
from habr_search.evaluation.ranking_metrics import RankedDocument, aggregate_name, evaluate_ranking
from habr_search.paths import DATA_DIR
from habr_search.retrieval.hybrid import HybridConfig, fuse_candidates
from habr_search.retrieval.router import ROUTER_VERSION, RouterConfig, RoutingDecision, RoutingHit, choose_route
from habr_search.retrieval.tantivy_index import DEFAULT_INDEX, TantivySearch


@dataclass(frozen=True)
class AutoReplay:
    ranking: tuple[RankedDocument, ...]
    decision: RoutingDecision


def replay_auto(context: ComparisonContext, records: Mapping[str, ComparisonRecord], titles: Mapping[int, str],
                fusion: HybridConfig = HybridConfig(dense_weight=4.0),
                router: RouterConfig = RouterConfig()) -> dict[str, AutoReplay]:
    """проверяем качество готовой политики, не подбирая её по ответам текущего набора."""
    if frozenset(titles) != context.universe or set(records) != {case.query_id for case in context.cases}:
        raise ValueError('нужны заголовки всего корпуса и завершённые выдачи всех запросов')
    result: dict[str, AutoReplay] = {}
    for case in context.cases:
        record = records[case.query_id]
        dense = [item.article_id for item in record.dense[:fusion.candidate_limit]]
        lexical = record.lexical[:fusion.candidate_limit]
        decision = choose_route(case.query, dense,
                                [RoutingHit(item.article_id, titles[item.article_id], item.score) for item in lexical], router)
        # модель сохраняет полное ранжирование; rrf работает только с выбранными кандидатами
        ranking = (record.dense if decision.engine == 'bge-m3' else
                   tuple(RankedDocument(item.article_id, item.score) for item in
                         fuse_candidates(dense, [item.article_id for item in lexical], fusion)))
        result[case.query_id] = AutoReplay(ranking, decision)
    return result


def write_report(context: ComparisonContext, records: Mapping[str, ComparisonRecord],
                 replayed: Mapping[str, AutoReplay], judgments: Mapping[str, Mapping[int, int]],
                 output: Path, cutoffs: Sequence[int] = (5, 10)) -> None:
    """старую задержку не выдаём за новую: офлайн-проверка измеряет только качество."""
    if output.exists():
        raise ValueError('папка отчёта уже существует; укажите новый --output')
    computed = {(engine, case.query_id): evaluate_ranking(
        replayed[case.query_id].ranking if engine == 'auto' else records[case.query_id].ranking(engine),
        judgments[case.query_id], context.universe, cutoffs=cutoffs,
        missing_score=None if engine == 'bge-m3' or (engine == 'auto' and
                           replayed[case.query_id].decision.engine == 'bge-m3') else 0.0,
    ) for engine in (*ENGINES, 'auto') for case in context.cases}
    metric_names = list(next((value.values for value in computed.values() if value.values), dict[str, float]()))
    output.mkdir(parents=True)
    with (output / 'per_query.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['engine', 'query_id', 'category', 'selected_engine', 'reason', 'features', *metric_names])
        for (engine, query_id), score in computed.items():
            case = next(case for case in context.cases if case.query_id == query_id)
            decision = replayed[query_id].decision
            writer.writerow([engine, query_id, case.category, decision.engine if engine == 'auto' else engine,
                             decision.reason if engine == 'auto' else '',
                             json.dumps(asdict(decision.features), ensure_ascii=False) if engine == 'auto' else '',
                             *(score.values.get(name, '') for name in metric_names)])
    with (output / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['engine', 'category', 'positive_queries', 'negative_queries', 'negative_empty',
                         *(aggregate_name(name) for name in metric_names)])
        for engine in (*ENGINES, 'auto'):
            for category in ('all', *sorted({case.category for case in context.cases})):
                scores = [computed[engine, case.query_id] for case in context.cases
                          if category == 'all' or case.category == category]
                positives = [score for score in scores if score.relevant_total]
                negatives = [score for score in scores if not score.relevant_total]
                values = [mean(score.values[name] for score in positives) if positives else '' for name in metric_names]
                writer.writerow([engine, category, len(positives), len(negatives),
                                 sum(score.returned == 0 for score in negatives), *values])
                if category == 'all':
                    print(engine + ': ' + '; '.join(f'{aggregate_name(name)}={value:.4f}'
                          for name, value in zip(metric_names, values)
                          if name in ('RR', 'Precision@5', 'Recall@10', 'nDCG@10') and isinstance(value, float)))


class CliArgs(argparse.Namespace):
    from_run: Path
    qrels: Path | None
    tantivy_path: Path
    output: Path | None
    candidate_limit: int
    dense_weight: float
    lexical_weight: float
    rrf_k: float


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', type=Path, required=True)
    parser.add_argument('--qrels', type=Path)
    parser.add_argument('--tantivy-path', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--candidate-limit', type=int, default=20)
    parser.add_argument('--dense-weight', type=float, default=4.0)
    parser.add_argument('--lexical-weight', type=float, default=1.0)
    parser.add_argument('--rrf-k', type=float, default=60.0)
    args = parser.parse_args(namespace=CliArgs())
    try:
        fusion = HybridConfig(args.candidate_limit, args.rrf_k, args.dense_weight, args.lexical_weight)
        if fusion.candidate_limit < 10:
            raise ValueError('для проверки @10 нужен candidate-limit не меньше 10')
        output = args.output or DATA_DIR / 'evaluation' / datetime.now(timezone.utc).strftime('auto-%Y%m%d-%H%M%S-%f')
        if output.exists():
            raise ValueError('папка отчёта уже существует; укажите новый --output')
        context = load_context(args.from_run)
        records = load_records(args.from_run, context)
        judgments = load_judgments(context, args.qrels)
        snapshot = TantivySearch(args.tantivy_path)
        if snapshot.info.revision != context.tantivy_revision:
            raise ValueError('версия tantivy отличается от сохранённого запуска')
        titles = snapshot.article_titles()
        router = RouterConfig()
        replayed = replay_auto(context, records, titles, fusion, router)
        write_report(context, records, replayed, judgments, output)
        provenance = {
            'router_version': ROUTER_VERSION, 'router': asdict(router), 'fusion': asdict(fusion),
            'timing_scope': 'not_measured_offline', 'lexical_availability': 'assumed_ready_no_timeout_simulation',
            'mws_calls': 0, 'source_run': str(args.from_run.resolve()),
            'context_sha256': hashlib.sha256((args.from_run / 'context.json').read_bytes()).hexdigest(),
            'rankings_sha256': hashlib.sha256((args.from_run / 'rankings.csv').read_bytes()).hexdigest(),
            'qrels_sha256': hashlib.sha256(args.qrels.read_bytes()).hexdigest() if args.qrels else None,
            'tantivy_revision': snapshot.info.revision, 'binary_positive_threshold': 'relevance > 0',
            'unjudged_policy': 'assumed_nonrelevant', 'label_status': 'partial' if any(
                frozenset(values) != context.universe for values in judgments.values()) else 'complete',
        }
        with (output / 'metadata.json').open('x', encoding='utf-8') as stream:
            json.dump(provenance, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        for case in context.cases:
            decision = replayed[case.query_id].decision
            print(f'{case.query_id}: {decision.engine}; {decision.reason}')
        print(f'отчёт: {output}; обращений к mws: 0; новая задержка не измерялась')
        if provenance['label_status'] == 'partial':
            print('разметка неполная; неразмеченные статьи считаются нерелевантными')
        return 0
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'проверка auto не завершена: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
