"""готовим общую таблицу разметки из уже сохранённого сравнения трёх поисков."""
from __future__ import annotations

import argparse
import csv
from contextlib import closing
from pathlib import Path

from habr_search.evaluation.evaluate import load_queries
from habr_search.evaluation.evaluate_hybrid import read_saved_report
from habr_search.evaluation.review import spreadsheet_text
from habr_search.ingestion.prepare import DEFAULT_SOURCE, PreparationStats, iter_prepared_articles
from habr_search.paths import PROJECT_ROOT


def prepare_pool(queries: Path, report: Path, source: Path, output: Path, depth: int = 20) -> int:
    """начальные положительные метки оставляем как seed; новые оценки заполняет проверяющий."""
    if output.exists() or depth < 1:
        raise ValueError('нужны свободное имя output и положительная глубина')
    cases = load_queries(queries)
    context, saved = read_saved_report(report, cases)
    if set(saved) != {case.query_id for case in cases} or depth > context.config.candidate_limit:
        raise ValueError('нужен завершённый отчёт с достаточной глубиной кандидатов')
    pools = {case.query_id: set((*saved[case.query_id].dense_ids[:depth], *saved[case.query_id].lexical_ids[:depth],
                                *saved[case.query_id].hybrid_ids[:depth], *case.relevant_ids)) for case in cases}
    needed = {identifier for pool in pools.values() for identifier in pool}
    with closing(iter_prepared_articles(source, PreparationStats())) as prepared:
        articles = {article.id: article for article in prepared if article.id in needed}
    if needed - articles.keys():
        raise ValueError('в корпусе отсутствуют статьи сохранённого отчёта')
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open('x', encoding='utf-8', newline='') as target:
        writer = csv.writer(target)
        writer.writerow(['query_id', 'category', 'query', 'article_id', 'title', 'url', 'text_sha256',
                         'bge_m3_rank', 'tantivy_rank', 'hybrid_rank', 'relevance', 'label_source', 'comment'])
        for case in cases:
            record = saved[case.query_id]
            ranks = [{identifier: rank for rank, identifier in enumerate(ids, 1)}
                     for ids in (record.dense_ids, record.lexical_ids, record.hybrid_ids)]
            for identifier in sorted(pools[case.query_id]):
                article = articles[identifier]
                seed = identifier in case.relevant_ids
                writer.writerow([case.query_id, case.category, spreadsheet_text(case.query), identifier,
                                 spreadsheet_text(article.title), article.url, article.text_hash,
                                 *(rank.get(identifier, '') for rank in ranks), 1 if seed else '',
                                 'seed_needs_review' if seed else '', ''])
                count += 1
    return count


class CliArgs(argparse.Namespace):
    queries: Path
    report: Path
    source: Path
    output: Path
    depth: int


def main() -> int:
    """используем существующие кандидаты, не вызывая модель и не меняя benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=PROJECT_ROOT / 'evaluation/benchmark_dev.csv')
    parser.add_argument('--report', type=Path, required=True, help='csv из evaluate_hybrid')
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--depth', type=int, default=20)
    args: CliArgs = parser.parse_args(namespace=CliArgs())
    try:
        count = prepare_pool(args.queries, args.report, args.source, args.output, args.depth)
        print(f'таблица: {args.output}; пар запрос—статья: {count}')
        print('relevance: 0 — не подходит; 1 — частично; 2 — подходит; 3 — прямо отвечает; пусто — не проверено')
        print('исходные метки требуют проверки; новых запросов к mws: 0')
        return 0
    except (ValueError, OSError, csv.Error) as exc:
        parser.exit(1, f'таблица не подготовлена: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
