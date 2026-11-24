"""сравниваем порядок rrf и реранкера на одних кандидатах работающего веб-поиска."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import requests

from habr_search.evaluation.comparison_data import load_context, load_judgments
from habr_search.evaluation.ranking_metrics import RankedDocument, evaluate_ranking
from habr_search.web.app import HealthResponse, SearchResponse

METRICS = ('Precision@5', 'nDCG@5', 'Recall@10', 'nDCG@10')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-run', type=Path, required=True)
    parser.add_argument('--qrels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='новый csv с результатами по запросам')
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    context = load_context(args.from_run)
    labels = load_judgments(context, args.qrels)
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    timings: list[float] = []
    with requests.Session() as session:
        health = session.get(args.server + '/api/health', timeout=10)
        health.raise_for_status()
        index = HealthResponse.model_validate(health.json())
        if (index.collection, index.articles, index.fragments) != (context.collection, len(context.universe), context.points):
            parser.exit(1, 'сервер использует другой корпус\n')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['query_id', 'engine', 'article_ids', *METRICS, 'rerank_ms', 'server_total_ms'])
            for case in context.cases:
                if case.category == 'no_answer':
                    continue
                reply = session.post(args.server + '/api/search', json={'query': case.query, 'top_k': 20}, timeout=330)
                reply.raise_for_status()
                result = SearchResponse.model_validate(reply.json())
                diag = result.diagnostics
                if diag is None or diag.rerank_status != 'applied':
                    parser.exit(1, f'{case.query_id}: реранкер не применился, проверьте журнал сервера\n')
                ids = tuple(item.article_id for item in result.results)
                if sorted(ids) != sorted(diag.rrf_ids):
                    raise ValueError('состав кандидатов изменился')
                for mode, order in [('rrf', diag.rrf_ids), ('reranker', ids)]:
                    # метрики на срезах используют только порядок статей
                    ranking = [RankedDocument(article_id, -rank) for rank, article_id in enumerate(order)]
                    values = evaluate_ranking(ranking, labels[case.query_id], context.universe,
                                              cutoffs=(5, 10), missing_score=-len(order)).values
                    for metric in METRICS:
                        totals[mode, metric].append(values[metric])
                    writer.writerow([case.query_id, mode, ';'.join(map(str, order)),
                                     *(values[m] for m in METRICS), diag.rerank_ms if mode == 'reranker' else '',
                                     result.elapsed_ms if mode == 'reranker' else ''])
                stream.flush()
                timings.append(diag.rerank_ms)
    for (mode, metric), samples in totals.items():
        print(f'{mode} {metric}: {mean(samples):.4f}')
    if timings:
        print(f'запросов: {len(timings)}, медиана времени реранкера: {median(timings):.0f} мс')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
