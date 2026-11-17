"""измеряем настоящий http-поиск разных режимов; каждый запрос заново вызывает mws."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter, sleep
from typing import TextIO

import requests

from habr_search.embeddings import read_api_key
from habr_search.evaluation.evaluate import EvaluationQuery, load_queries
from habr_search.paths import DATA_DIR, PROJECT_ROOT
from habr_search.web.app import SearchResponse
from habr_search.retrieval.ltr import LearnedRanker

ENGINES: tuple[str, ...] = ('bge-m3', 'hybrid', 'auto')
COLUMNS: tuple[str, ...] = ('query_id', 'engine', 'repeat', 'status', 'http_ms', 'server_ms',
                           'selected_engine', 'route_reason', 'lexical_status', 'article_ids', 'error_type')


@dataclass(frozen=True)
class Timing:
    engine: str
    http_ms: float


def collect_timings(session: requests.Session, addresses: Mapping[str, str], queries: Sequence[EvaluationQuery],
                    repeats: int, top_k: int, timeout: float, stream: TextIO, *,
                    engines: Sequence[str] = ENGINES) -> list[Timing]:
    """чередуем режимы и сразу сохраняем каждую попытку; платные post не повторяются."""
    if repeats < 1 or not 1 <= top_k <= 20 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('нужны положительные repeats/timeout и top-k от 1 до 20')
    writer = csv.writer(stream)
    writer.writerow(COLUMNS)
    stream.flush()
    timings: list[Timing] = []
    order_template = tuple(engines)
    if not order_template or len(set(order_template)) != len(order_template) or set(order_template) != set(addresses):
        raise ValueError('нужны разные режимы с соответствующими адресами')
    for repeat in range(repeats):
        for index, query in enumerate(queries):
            shift = (index + repeat) % len(order_template)
            order = order_template[shift:] + order_template[:shift]
            for engine in order:
                start = perf_counter()
                try:
                    response = session.post(addresses[engine] + '/api/search',
                                            json={'query': query.query, 'top_k': top_k}, timeout=timeout)
                    response.raise_for_status()
                    payload = SearchResponse.model_validate(response.json())
                except Exception as exc:
                    elapsed = (perf_counter() - start) * 1000
                    writer.writerow([query.query_id, engine, repeat + 1, 'error', elapsed,
                                     '', '', '', '', '', type(exc).__name__])
                    stream.flush()
                    raise RuntimeError('замер остановлен после ошибки; автоматического повтора нет; '
                                       'последняя попытка могла быть оплачена') from exc
                elapsed = (perf_counter() - start) * 1000
                diagnostic = payload.diagnostics
                writer.writerow([query.query_id, engine, repeat + 1, 'ok', elapsed, payload.elapsed_ms,
                                 diagnostic.decision.engine if diagnostic is not None else engine,
                                 diagnostic.decision.reason if diagnostic is not None else '',
                                 diagnostic.lexical_status if diagnostic is not None else '',
                                 ';'.join(str(item.article_id) for item in payload.results), ''])
                stream.flush()
                timings.append(Timing(engine, elapsed))
                print(f'{query.query_id}: {engine}; {elapsed:.0f} мс', flush=True)
    return timings


def stop_server(process: subprocess.Popen[bytes]) -> None:
    """останавливаем только запущенные этой командой процессы, не чужие серверы."""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_ready(session: requests.Session, url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = perf_counter() + 45
    while perf_counter() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'сервер {url} завершился до готовности; проверьте его сообщение в терминале')
        try:
            response = session.get(url + '/api/health', timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass
        sleep(0.1)
    raise RuntimeError(f'сервер {url} не стал готов за 45 секунд')


class CliArgs(argparse.Namespace):
    queries: Path
    output: Path
    limit: int
    repeats: int
    top_k: int
    base_port: int
    timeout: float
    qdrant_url: str
    tantivy_path: Path
    ranker_model: Path | None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=PROJECT_ROOT / 'evaluation/queries_auto.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=6, help='первые запросы набора; 0 означает весь набор')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=20, help='как у веб-страницы с кнопкой показать ещё')
    parser.add_argument('--base-port', type=int, default=8030)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--qdrant-url', default='http://127.0.0.1:6333')
    parser.add_argument('--tantivy-path', type=Path, default=DATA_DIR / 'tantivy')
    parser.add_argument('--ranker-model', type=Path, help='добавить четвёртый режим ltr с обученной моделью')
    args = parser.parse_args(namespace=CliArgs())
    try:
        if args.output.exists():
            raise ValueError('папка замера уже существует; укажите новую --output')
        engines = (*ENGINES, 'ltr') if args.ranker_model is not None else ENGINES
        if (args.limit < 0 or args.repeats < 1 or not 1 <= args.top_k <= 20
                or not 1 <= args.base_port <= 65536 - len(engines) or not math.isfinite(args.timeout) or args.timeout <= 0):
            raise ValueError('некорректные limit/repeats/top-k/base-port/timeout')
        if args.ranker_model is not None:
            LearnedRanker(args.ranker_model)  # проверяем артефакт до ключа и старта процессов
        queries = load_queries(args.queries)
        selected = queries[:args.limit] if args.limit else queries
        # порты проверяем до чтения ключа; занятые процессы не завершаем
        with ExitStack() as ports:
            for port in range(args.base_port, args.base_port + len(engines)):
                listener = ports.enter_context(socket.socket())
                listener.bind(('127.0.0.1', port))
        planned = len(selected) * args.repeats * len(engines)
        print(f'план: {planned} новых поисковых запросов к mws; прогрева и автоматических повторов нет', flush=True)
        key = read_api_key()
        args.output.mkdir(parents=True)
        metadata: dict[str, object] = {
            'timing_scope': 'real_loopback_http_end_to_end_with_live_mws', 'complete': False,
            'queries_sha256': hashlib.sha256(args.queries.read_bytes()).hexdigest(),
            'query_ids': [query.query_id for query in selected], 'repeats': args.repeats,
            'top_k': args.top_k, 'planned_mws_requests': planned, 'warmups': 0,
            'engine_order': 'rotated_per_query_and_repeat', 'automatic_retries': False,
            'engines': list(engines), 'ranker_model': str(args.ranker_model.resolve()) if args.ranker_model else None,
            'note': 'small_sample_no_load; p95_with_six_samples_is_maximum; models_called_independently',
        }
        metadata_path = args.output / 'metadata.json'
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        with ExitStack() as resources:
            session = resources.enter_context(requests.Session())
            session.trust_env = False  # локальный http не отправляем через прокси окружения
            addresses: dict[str, str] = {}
            environment = dict(os.environ, MWS_API_KEY=key)
            for offset, engine in enumerate(engines):
                port = args.base_port + offset
                command = [sys.executable, '-m', 'habr_search.web.app', '--engine', engine,
                           '--qdrant-url', args.qdrant_url, '--tantivy-path', str(args.tantivy_path),
                           '--port', str(port), '--timeout', str(args.timeout), '--without-spelling']
                if engine == 'ltr' and args.ranker_model is not None:
                    command += ['--ranker-model', str(args.ranker_model.resolve())]
                process: subprocess.Popen[bytes] = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                resources.callback(stop_server, process)
                addresses[engine] = f'http://127.0.0.1:{port}'
                wait_ready(session, addresses[engine], process)
            with (args.output / 'requests.csv').open('x', encoding='utf-8', newline='') as stream:
                timings = collect_timings(session, addresses, selected, args.repeats,
                                           args.top_k, args.timeout + 10, stream, engines=engines)
        with (args.output / 'summary.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['engine', 'requests', 'median_http_ms', 'p95_http_ms'])
            for engine in engines:
                times = sorted(timing.http_ms for timing in timings if timing.engine == engine)
                p95 = times[math.ceil(len(times) * 0.95) - 1]
                writer.writerow([engine, len(times), median(times), p95])
                print(f'{engine}: n={len(times)}; медиана {median(times):.0f} мс; p95 {p95:.0f} мс')
        metadata['complete'] = True
        metadata['successful_http_requests'] = len(timings)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        print(f'замер: {args.output}; запросов поиска: {len(timings)}; ключ в файлы не записывался')
        return 0
    except KeyboardInterrupt:
        print('\nзамер остановлен; незавершённая попытка могла быть оплачена; автоматического повтора нет')
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f'замер не завершён: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
