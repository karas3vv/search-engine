"""сравниваем bge-m3, tantivy и гибрид; сохранённые выдачи пересчитываем без mws."""
from __future__ import annotations

import argparse
import csv
import math
import os
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from qdrant_client import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from habr_search.embeddings import DEFAULT_URL, EmbeddingConfig, EmbeddingsClient, read_api_key
from habr_search.evaluation.comparison_data import (
    ComparisonContext, ComparisonRecord, load_context, load_judgments, load_records,
)
from habr_search.evaluation.comparison_metrics import create_metrics
from habr_search.evaluation.evaluate import EvaluationQuery, load_queries, prepare_queries
from habr_search.evaluation.ranking_metrics import RankedDocument, validate_ranking
from habr_search.ingestion.chunking import ChunkingConfig, load_tokenizer
from habr_search.ingestion.index import EmbeddingProvider, collection_name
from habr_search.model_profiles import BGE_M3, deployment_name
from habr_search.paths import DATA_DIR, PROJECT_ROOT
from habr_search.retrieval.hybrid import HybridConfig, HybridRetriever, fuse_candidates
from habr_search.retrieval.tantivy_index import DEFAULT_INDEX, TantivySearch, literal_query
from habr_search.vector_store import connect_store, resolve_qdrant_config


def full_dense_ranking(retriever: HybridRetriever, vector: list[float], universe: frozenset[int]) -> tuple[RankedDocument, ...]:
    """для pr-auc получаем оценки всех статей, не передавая тексты и отрывки."""
    if len(vector) != BGE_M3.vector_size or not all(math.isfinite(v) for v in vector) or not any(vector):
        raise ValueError('нужен конечный ненулевой вектор bge-m3 размерности 1024')
    response = retriever.store.query_points_groups(
        collection_name=retriever.name, query=vector, group_by='article_id', group_size=1,
        limit=len(universe), with_payload=['article_id'], with_vectors=False,
        search_params=models.SearchParams(exact=True),
    )
    found: list[RankedDocument] = []
    for group in response.groups:
        if len(group.hits) != 1:
            raise ValueError('qdrant не вернул лучший фрагмент статьи')
        hit = group.hits[0]
        identifier: object = (hit.payload or {}).get('article_id')
        if type(identifier) is not int:
            raise ValueError('в qdrant отсутствует id статьи')
        found.append(RankedDocument(identifier, float(hit.score)))
    result = tuple(sorted(found, key=lambda d: (-d.score, d.article_id)))
    validate_ranking(result, universe)
    if frozenset(d.article_id for d in result) != universe:
        raise ValueError('qdrant вернул не все статьи; pr-auc по обрезанной выдаче не рассчитывается')
    return result


def collect_rankings(
    prepared: Sequence[tuple[EvaluationQuery, str]], retriever: HybridRetriever,
    provider: EmbeddingProvider, universe: frozenset[int],
) -> Iterator[ComparisonRecord]:
    """один эмбеддинг обслуживает модель и гибрид; время относится к полному ранжированию."""
    for case, text in prepared:
        retriever.check_context()
        start = perf_counter()
        vector = provider.embed([text])[0]
        embedding_ms = (perf_counter() - start) * 1000
        stage = perf_counter()
        dense = full_dense_ranking(retriever, vector, universe)
        dense_ms = embedding_ms + (perf_counter() - stage) * 1000
        stage = perf_counter()
        lexical = tuple(RankedDocument(i, score) for i, score in retriever.lexical.ranked_matches(case.query))
        lexical_ms = (perf_counter() - stage) * 1000
        validate_ranking(lexical, universe)
        limit = retriever.config.candidate_limit
        hybrid = tuple(RankedDocument(item.article_id, item.score) for item in fuse_candidates(
            [d.article_id for d in dense[:limit]], [d.article_id for d in lexical[:limit]], retriever.config,
        ))
        elapsed = (perf_counter() - start) * 1000
        retriever.check_context()
        yield ComparisonRecord(case.query_id, datetime.now(timezone.utc).isoformat(),
                               dense, lexical, hybrid, dense_ms, lexical_ms, elapsed)


class CliArgs(argparse.Namespace):
    """явные типы параметров сбора, продолжения и офлайн-пересчёта."""
    queries: Path
    output: Path | None
    report_output: Path | None
    from_run: Path | None
    resume: Path | None
    qrels: Path | None
    qdrant_url: str | None
    qdrant_path: Path | None
    tantivy_path: Path
    tokenizer_file: Path | None
    candidate_limit: int
    rrf_k: float
    dense_weight: float
    lexical_weight: float
    top_k: int
    cutoffs: list[int]
    pool_depth: int
    timeout: float
    max_tokens: int
    overlap_tokens: int
    check_only: bool


def timestamp() -> str:
    """отдельный суффикс позволяет сохранять несколько пересчётов одного запуска."""
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')


def main() -> int:
    """отделяем платный сбор ранжирований от бесплатного пересчёта по новой разметке."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=PROJECT_ROOT / 'evaluation/benchmark_dev.csv')
    parser.add_argument('--output', type=Path, help='новая папка полного запуска')
    parser.add_argument('--report-output', type=Path, help='новая папка таблиц метрик')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--from-run', type=Path, help='пересчитать готовый запуск без mws и индексов')
    source.add_argument('--resume', type=Path, help='продолжить незавершённый запуск в той же папке')
    parser.add_argument('--qrels', type=Path, help='csv с query_id, article_id, relevance; расширяет исходные метки')
    location = parser.add_mutually_exclusive_group()
    location.add_argument('--qdrant-url')
    location.add_argument('--qdrant-path', type=Path)
    parser.add_argument('--tantivy-path', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--tokenizer-file', type=Path)
    parser.add_argument('--candidate-limit', type=int, default=20)
    parser.add_argument('--rrf-k', type=float, default=60.0)
    parser.add_argument('--dense-weight', type=float, default=1.0)
    parser.add_argument('--lexical-weight', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=5, help='размер выдачи для precision, recall и f1 без суффикса')
    parser.add_argument('--cutoffs', nargs='+', type=int, default=[1, 3, 5, 10, 20])
    parser.add_argument('--pool-depth', type=int, default=20, help='глубина кандидатов каждого режима для разметки')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--overlap-tokens', type=int, default=128)
    parser.add_argument('--check-only', action='store_true', help='проверить входы и индексы без ключа и mws')
    args = parser.parse_args(namespace=CliArgs())
    try:
        if (not 1 <= args.top_k <= 20 or any(not 1 <= k <= 20 for k in args.cutoffs)
                or len(set(args.cutoffs)) != len(args.cutoffs) or args.pool_depth < 1):
            raise ValueError('нужны разные k от 1 до 20 и положительный pool-depth')
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError('timeout должен быть положительным конечным числом')
        if args.report_output is not None and args.report_output.exists():
            raise ValueError('папка метрик уже существует')
        if args.from_run is not None:
            if args.check_only or args.output is not None:
                raise ValueError('from-run используется без check-only и output; задайте report-output')
            context = load_context(args.from_run)
            cached_records = load_records(args.from_run, context)
            destination = args.report_output or args.from_run / f'metrics-{timestamp()}'
            create_metrics(context, cached_records, destination, args.top_k, args.cutoffs, args.qrels, args.pool_depth)
            print('новых запросов к mws: 0; исходный запуск не изменялся')
            return 0
        config = HybridConfig(args.candidate_limit, args.rrf_k, args.dense_weight, args.lexical_weight)
        if args.candidate_limit < max(args.top_k, *args.cutoffs):
            raise ValueError('candidate-limit не должен быть меньше выбранных k')
        if args.resume is not None and args.output is not None:
            raise ValueError('resume уже указывает папку запуска; output не требуется')
        output = args.resume or args.output or DATA_DIR / 'evaluation' / f'comparison-{timestamp()}'
        if args.resume is None and output.exists():
            raise ValueError('папка запуска уже существует; используйте resume или новый output')
        cases = tuple(load_queries(args.queries))
        for case in cases:
            literal_query(case.query)
        url, deployment = os.environ.get('MWS_EMBEDDINGS_URL', DEFAULT_URL), deployment_name(BGE_M3)
        name = collection_name(url, deployment, ChunkingConfig(args.max_tokens, args.overlap_tokens), BGE_M3)
        with ExitStack() as resources:
            store = connect_store(resolve_qdrant_config(args.qdrant_url, args.qdrant_path))
            resources.callback(store.close)
            retriever = HybridRetriever(store, name, TantivySearch(args.tantivy_path), config)
            universe = retriever.lexical.article_ids()
            if not {i for q in cases for i in q.relevant_ids} <= universe:
                raise ValueError('в индексе отсутствуют эталонные статьи')
            context = ComparisonContext(name, retriever.revision, retriever.lexical.info.revision,
                                        retriever.points, universe, cases, config)
            load_judgments(context, args.qrels)  # ошибки разметки выявляем до платного api
            records: dict[str, ComparisonRecord] = {}
            if args.resume is not None:
                if load_context(output) != context:
                    raise ValueError('запросы, настройки или версии индексов изменились после сохранения')
                records = load_records(output, context)
            pending = [case for case in cases if case.query_id not in records]
            prepared = prepare_queries(pending, load_tokenizer(args.tokenizer_file, BGE_M3), BGE_M3) if pending else []
            print(f'запросов: {len(cases)}; статей: {len(universe)}; сохранено: {len(records)}; осталось: {len(pending)}')
            if args.check_only:
                print('проверка завершена; mws не вызывался; файлы не создавались')
                return 0
            if args.resume is None:
                context.save(output)
            print(f'запуск: {output}', flush=True)
            if pending:
                provider = resources.enter_context(closing(EmbeddingsClient(EmbeddingConfig(read_api_key(), url, deployment, args.timeout))))
                with (output / 'rankings.csv').open('a', encoding='utf-8', newline='') as stream:
                    writer = csv.writer(stream)
                    for record in collect_rankings(prepared, retriever, provider, universe):
                        writer.writerow(record.row())
                        stream.flush()
                        records[record.query_id] = record
                        print(f'{len(records)}/{len(cases)} {record.query_id}', flush=True)
            retriever.check_context()
        destination = args.report_output or output / f'metrics-{timestamp()}'
        create_metrics(context, records, destination, args.top_k, args.cutoffs, args.qrels, args.pool_depth)
        print(f'новых запросов к mws: {len(pending)}; число режимов: 3')
        return 0
    except KeyboardInterrupt:
        print('\nостановлено; сохранённые запросы можно продолжить через --resume')
        return 130
    except (UnexpectedResponse, ResponseHandlingException):
        parser.exit(1, 'не удалось прочитать qdrant; проверьте контейнер и адрес\n')
    except (ValueError, OSError, RuntimeError, csv.Error) as exc:
        parser.exit(1, f'оценка не завершена: {exc}\nзавершённые ранжирования сохранены в папке запуска\n')


if __name__ == '__main__':
    raise SystemExit(main())
