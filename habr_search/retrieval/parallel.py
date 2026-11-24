"""перекрываем локальный поиск ожиданием mws и ограничиваем очередь tantivy."""
from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from habr_search.ingestion.index import EmbeddingProvider
from habr_search.model_profiles import BGE_M3
from habr_search.retrieval.hybrid import FusedRank, HybridResult, HybridRetriever, fuse_rankings
from habr_search.retrieval.rerank import Reranker, RerankError
from habr_search.retrieval.router import (
    RouterConfig, RoutingDecision, RoutingHit, choose_route,
)
from habr_search.retrieval.search import search_articles
from habr_search.retrieval.tantivy_index import LexicalCandidate, literal_query

LOGGER: logging.Logger = logging.getLogger('habr_search.parallel')
LexicalStatus = Literal['ready', 'timeout', 'busy', 'error']


@dataclass(frozen=True)
class LexicalRun:
    candidates: tuple[LexicalCandidate, ...]
    elapsed_ms: float


@dataclass(frozen=True)
class ParallelDiagnostics:
    decision: RoutingDecision
    lexical_status: LexicalStatus
    embedding_ms: float
    qdrant_ms: float
    lexical_ms: float | None
    routing_ms: float
    snippets_ms: float
    dense_weight: float
    lexical_weight: float
    lexical_wait_ms: float | None  # в обычном гибриде ждём обе ветки

    rerank_status: Literal['off', 'applied', 'skipped', 'error'] = 'off'
    rerank_ms: float = 0.0
    rrf_ids: tuple[int, ...] = ()  # исходный порядок для парной оценки на тех же кандидатах


@dataclass(frozen=True)
class ParallelRun:
    results: tuple[HybridResult, ...]
    diagnostics: ParallelDiagnostics


class ParallelRetriever:
    """один запрос сервиса использует один вызов mws и не больше одного локального задания."""
    def __init__(self, hybrid: HybridRetriever, embeddings: EmbeddingProvider,
                 router: RouterConfig | None = None, reranker: Reranker | None = None) -> None:
        self.reranker = reranker
        self.hybrid: HybridRetriever = hybrid
        self.embeddings: EmbeddingProvider = embeddings
        self.router: RouterConfig | None = router
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tantivy-search')
        self.pending: Future[LexicalRun] | None = None

    def close(self) -> None:
        """при остановке дожидаемся текущего локального чтения и освобождаем поток."""
        self.executor.shutdown(wait=True, cancel_futures=True)

    def _lexical(self, query: str) -> LexicalRun:
        start = perf_counter()
        found = self.hybrid.lexical.select_candidates(query, self.hybrid.config.candidate_limit)
        return LexicalRun(tuple(found), (perf_counter() - start) * 1000)

    def search(self, query: str, prepared: str, top_k: int, *, prechecked: bool = False) -> ParallelRun:
        """вызывается под блокировкой сервиса; qdrant и клиент mws остаются в его потоке."""
        literal_query(query)
        if not 1 <= top_k <= min(20, self.hybrid.config.candidate_limit):
            raise ValueError('top-k должен быть от 1 до 20 и не больше candidate-limit')
        if not prechecked:
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
        wait_ms = self.router.lexical_wait_ms if self.router is not None else None
        if future is not None:
            try:
                local = future.result(timeout=wait_ms / 1000 if wait_ms is not None else None)
                lexical, lexical_ms = local.candidates, local.elapsed_ms
            except TimeoutError:
                status = 'timeout'
                future.cancel()
            except Exception as exc:
                # сбой необязательной ветки не скрывает ошибку модели или qdrant
                status = 'error'
                LOGGER.warning('lexical_search_error type=%s', type(exc).__name__)
        start = perf_counter()
        dense_ids = [item.article_id for item in dense]
        if status != 'ready':
            decision = RoutingDecision('bge-m3', f'lexical_{status}')
        elif self.router is None:
            decision = RoutingDecision('hybrid', 'rrf', version='rrf-v1')
        else:
            hits = [RoutingHit(item.article_id, item.title, item.score) for item in lexical]
            decision = choose_route(query, dense_ids, hits, self.router)
        # в режиме модели сохраняем её косинусные оценки и семантические отрывки
        dense_ranks = [FusedRank(item.article_id, item.score, index, None)
                       for index, item in enumerate(dense[:top_k], 1)]
        if decision.engine == 'bge-m3':
            ranks = dense_ranks
        else:
            depth = max(top_k, self.reranker.depth) if self.reranker else top_k
            ranks = fuse_rankings(dense_ids, [item.article_id for item in lexical], depth, self.hybrid.config)
        routing_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        try:
            results = self.hybrid.render(query, dense, lexical if decision.engine == 'hybrid' else (), ranks)
        except Exception as exc:
            if decision.engine != 'hybrid':
                raise
            status = 'error'
            decision = RoutingDecision('bge-m3', 'lexical_snippet_error', decision.features, decision.version)
            LOGGER.warning('lexical_snippet_error type=%s', type(exc).__name__)
            results = self.hybrid.render(query, dense, (), dense_ranks)
        snippets_ms = (perf_counter() - start) * 1000
        self.hybrid.check_context()
        rerank_status: Literal['off', 'applied', 'skipped', 'error'] = 'off'
        rerank_ms = 0.0
        rrf_ids: tuple[int, ...] = ()
        if self.reranker is not None:
            rerank_status = 'skipped'
            if decision.engine == 'hybrid' and len(results) > 1:
                rrf_ids = tuple(item.article_id for item in results)
                start = perf_counter()
                try:
                    results = self.reranker.rank(query, results)
                    rerank_status = 'applied'
                except RerankError as exc:
                    rerank_status = 'error'
                    LOGGER.warning('rerank_error: %s', exc)
                rerank_ms = (perf_counter() - start) * 1000
        diagnostics = ParallelDiagnostics(decision, status, embedding_ms, qdrant_ms, lexical_ms,
                                          routing_ms, snippets_ms, self.hybrid.config.dense_weight,
                                          self.hybrid.config.lexical_weight if decision.engine == 'hybrid' else 0.0,
                                          wait_ms, rerank_status, rerank_ms, rrf_ids)
        # журнал не содержит исходного запроса, ключей, текстов статей или векторов
        LOGGER.info('parallel_search route=%s reason=%s version=%s lexical_status=%s '
                    'embedding_ms=%.1f qdrant_ms=%.1f lexical_ms=%s routing_ms=%.1f snippets_ms=%.1f '
                    'weights=%g:%g coverage=%.3f gap=%.3f overlap=%d',
                    decision.engine, decision.reason, decision.version, status, embedding_ms, qdrant_ms,
                    round(lexical_ms, 1) if lexical_ms is not None else None, routing_ms, snippets_ms,
                    diagnostics.dense_weight, diagnostics.lexical_weight, decision.features.title_coverage,
                    decision.features.lexical_gap, decision.features.top5_overlap)
        return ParallelRun(tuple(results[:top_k]), diagnostics)
