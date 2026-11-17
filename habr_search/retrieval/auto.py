"""перекрываем локальный поиск ожиданием mws и ограничиваем очередь tantivy."""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from habr_search.ingestion.index import EmbeddingProvider
from habr_search.model_profiles import BGE_M3
from habr_search.retrieval.hybrid import FusedRank, HybridResult, HybridRetriever
from habr_search.retrieval.router import (
    RouterConfig, RoutingDecision, RoutingHit, choose_route, route_ranking,
)
from habr_search.retrieval.search import search_articles
from habr_search.retrieval.tantivy_index import LexicalCandidate, literal_query

LOGGER: logging.Logger = logging.getLogger('habr_search.auto')
LexicalStatus = Literal['ready', 'timeout', 'busy', 'error']


@dataclass(frozen=True)
class LexicalRun:
    candidates: tuple[LexicalCandidate, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class AutoDiagnostics:
    decision: RoutingDecision
    lexical_status: LexicalStatus
    embedding_ms: float
    qdrant_ms: float
    lexical_ms: float | None
    routing_ms: float
    snippets_ms: float
    dense_weight: float
    lexical_weight: float
    lexical_wait_ms: float


@dataclass(frozen=True)
class AutoRun:
    results: tuple[HybridResult, ...]
    diagnostics: AutoDiagnostics


class AutoRetriever:
    """один запрос сервиса использует один вызов mws и не больше одного локального задания."""
    def __init__(self, hybrid: HybridRetriever, embeddings: EmbeddingProvider,
                 config: RouterConfig = RouterConfig()) -> None:
        self.hybrid: HybridRetriever = hybrid
        self.embeddings: EmbeddingProvider = embeddings
        self.config: RouterConfig = config
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tantivy-auto')
        self.pending: Future[LexicalRun] | None = None

    def close(self) -> None:
        """при остановке дожидаемся текущего локального чтения и освобождаем поток."""
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _lexical(self, query: str) -> LexicalRun:
        start = perf_counter()
        found = self.hybrid.lexical.select_candidates(query, self.hybrid.config.candidate_limit)
        return LexicalRun(tuple(found), (perf_counter() - start) * 1000)

    def search(self, query: str, prepared: str, top_k: int) -> AutoRun:
        """вызывается под блокировкой сервиса; qdrant и клиент mws остаются в его потоке."""
        literal_query(query)
        if not 1 <= top_k <= min(20, self.hybrid.config.candidate_limit):
            raise ValueError('top-k должен быть от 1 до 20 и не больше candidate-limit')
        self.hybrid.check_context()
        future: Future[LexicalRun] | None = None
        if self.pending is None or self.pending.done():
            future = self.executor.submit(self._lexical, query)
            self.pending = future
        # если прошлое чтение ещё выполняется, новое не ставится в очередь
        status: LexicalStatus = 'busy' if future is None else 'ready'
        try:
            start = perf_counter()
            vector = self.embeddings.embed([prepared])[0]
            embedding_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            dense = search_articles(self.hybrid.store, self.hybrid.name, vector,
                                    self.hybrid.config.candidate_limit, BGE_M3.vector_size)
            qdrant_ms = (perf_counter() - start) * 1000
        except BaseException:
            if future is not None:
                future.cancel()  # выполняющееся rust-чтение не прерывается принудительно
            raise
        lexical: tuple[LexicalCandidate, ...] = ()
        lexical_ms: float | None = None
        if future is not None:
            try:
                local = future.result(timeout=self.config.lexical_wait_ms / 1000)
                lexical, lexical_ms = local.candidates, local.elapsed_ms
            except TimeoutError:
                status = 'timeout'
                future.cancel()
            except Exception as exc:
                # сбой необязательной ветки не скрывает ошибку модели или qdrant
                status = 'error'
                LOGGER.warning('auto_lexical_error type=%s', type(exc).__name__)
        start = perf_counter()
        dense_ids = [item.article_id for item in dense]
        decision = (choose_route(query, dense_ids,
                                [RoutingHit(item.article_id, item.title, item.score) for item in lexical], self.config)
                    if status == 'ready' else RoutingDecision('bge-m3', f'lexical_{status}'))
        ids = route_ranking(dense_ids, [item.article_id for item in lexical], decision, self.hybrid.config)[:top_k]
        dense_by_id = {item.article_id: item for item in dense}
        # в режиме модели сохраняем её косинусные оценки и семантические отрывки
        if decision.engine == 'bge-m3':
            ranks = [FusedRank(identifier, dense_by_id[identifier].score, index, None)
                     for index, identifier in enumerate(ids, 1)]
        else:
            from habr_search.retrieval.hybrid import fuse_rankings
            ranks = fuse_rankings(dense_ids, [item.article_id for item in lexical], top_k, self.hybrid.config)
        routing_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        try:
            results = self.hybrid.render(query, dense, lexical if decision.engine == 'hybrid' else (), ranks)
        except Exception as exc:
            if decision.engine != 'hybrid':
                raise
            status = 'error'
            decision = RoutingDecision('bge-m3', 'lexical_snippet_error', decision.features)
            LOGGER.warning('auto_snippet_error type=%s', type(exc).__name__)
            ranks = [FusedRank(item.article_id, item.score, index, None)
                     for index, item in enumerate(dense[:top_k], 1)]
            results = self.hybrid.render(query, dense, (), ranks)
        snippets_ms = (perf_counter() - start) * 1000
        self.hybrid.check_context()
        diagnostics = AutoDiagnostics(decision, status, embedding_ms, qdrant_ms, lexical_ms,
                                      routing_ms, snippets_ms, self.hybrid.config.dense_weight,
                                      self.hybrid.config.lexical_weight if decision.engine == 'hybrid' else 0.0,
                                      self.config.lexical_wait_ms)
        # журнал не содержит исходного запроса, ключей, текстов статей или векторов
        LOGGER.info('auto_search route=%s reason=%s version=%s lexical_status=%s '
                    'embedding_ms=%.1f qdrant_ms=%.1f lexical_ms=%s routing_ms=%.1f snippets_ms=%.1f '
                    'weights=%g:%g coverage=%.3f gap=%.3f overlap=%d',
                    decision.engine, decision.reason, decision.version, status, embedding_ms, qdrant_ms,
                    round(lexical_ms, 1) if lexical_ms is not None else None, routing_ms, snippets_ms,
                    diagnostics.dense_weight, diagnostics.lexical_weight, decision.features.title_coverage,
                    decision.features.lexical_gap, decision.features.top5_overlap)
        return AutoRun(tuple(results), diagnostics)
