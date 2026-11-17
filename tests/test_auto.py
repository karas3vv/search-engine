"""проверяем маршрутизацию, параллельность и деградацию без платного api."""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import unittest
from contextlib import closing, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from habr_search.embeddings import DEFAULT_URL, EmbeddingAPIError
from habr_search.evaluation.comparison_data import ComparisonContext, ComparisonRecord, COLUMNS
from habr_search.evaluation.evaluate import EvaluationQuery
from habr_search.evaluation.evaluate_auto import main as evaluate_main, replay_auto
from habr_search.evaluation.ranking_metrics import RankedDocument
from habr_search.ingestion.chunking import ChunkingConfig, chunk_article
from habr_search.ingestion.index import collection_name, ensure_collection, index_chunks
from habr_search.model_profiles import BGE_M3
from habr_search.retrieval.hybrid import HybridConfig, fuse_candidates, fuse_rankings
from habr_search.retrieval.router import RouterConfig, RoutingHit, choose_route, query_identifiers
from habr_search.retrieval.search import SearchResult
from habr_search.retrieval.tantivy_index import LexicalCandidate, LexicalResult, build_index
from habr_search.web.app import AutoService, TantivyService, create_app, main as web_main
from tests.test_chunking import CharacterTokenizer, article
from tests.test_evaluate import FakeEmbeddings


class RouterTests(unittest.TestCase):
    def test_semantic_queries_do_not_change_dense_ranking(self) -> None:
        decision = choose_route('как готовить документы из таблиц', [1, 2],
                                [RoutingHit(2, 'Обработка таблиц', 20)])
        self.assertEqual(decision.engine, 'bge-m3')
        self.assertEqual(decision.reason, 'semantic_query')

    def test_exact_identifier_can_promote_lexical_answer(self) -> None:
        decision = choose_route('как использовать ExecuteExcel4Macro', [1, 2],
                                [RoutingHit(2, 'Пример ExecuteExcel4Macro', 20)])
        self.assertEqual(decision.engine, 'hybrid')
        self.assertEqual(decision.features.title_coverage, 1.0)
        self.assertEqual(fuse_rankings([1, 2], [2], 2, HybridConfig(dense_weight=4))[0].article_id, 2)
        self.assertEqual(query_identifiers('ERR_CONNECTION_RESET'), ('err_connection_reset',))
        self.assertEqual(query_identifiers('Power Automate'), ())
        quoted = choose_route('как использовать "перенос данных"', [1, 2],
                              [RoutingHit(2, 'Перенос данных', 5)])
        self.assertEqual(quoted.engine, 'hybrid')

    def test_ambiguous_or_partial_matches_keep_model(self) -> None:
        cases: list[tuple[str, list[int], list[RoutingHit], str]] = [
            ('python3', [1], [RoutingHit(2, 'python310', 20)], 'identifier_not_in_top_title'),
            ('ExecuteExcel4Macro', [2, 1], [RoutingHit(2, 'ExecuteExcel4Macro', 20)], 'exact_answer_already_first'),
            ('ExecuteExcel4Macro', [1], [RoutingHit(2, 'ExecuteExcel4Macro', 20),
                                     RoutingHit(3, 'ExecuteExcel4Macro', 20)], 'ambiguous_lexical_results'),
            ('ExecuteExcel4Macro перенос документов сервер база', [1],
             [RoutingHit(2, 'ExecuteExcel4Macro', 20)], 'insufficient_title_coverage'),
            ('ExecuteExcel4Macro', [1], [], 'no_lexical_matches'),
        ]
        for query, dense, lexical, reason in cases:
            with self.subTest(reason=reason):
                decision = choose_route(query, dense, lexical)
                self.assertEqual(decision.engine, 'bge-m3')
                self.assertEqual(decision.reason, reason)
        for value in (-1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                RouterConfig(lexical_wait_ms=value)
        with self.assertRaises(ValueError):
            RouterConfig(min_title_coverage=2)


class AutoTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index = self.root / 'tantivy'
        source = self.root / 'corpus.sqlite3'
        self.documents = [article('обработка документов общими способами', 'Другие документы', 1),
                          article('обработка данных закрытой книги', 'ExecuteExcel4Macro', 2),
                          article('обработка звонков', 'Телефония', 3)]
        with sqlite3.connect(source) as connection:
            connection.execute('CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
            for document in self.documents:
                connection.execute('INSERT INTO articles VALUES (?,?)', (document.id, json.dumps({
                    'id': document.id, 'title': document.title, 'url': document.url, 'text': document.text})))
        build_index(source, self.index)
        self.store = QdrantClient(':memory:')
        self.addCleanup(self.store.close)
        self.name = collection_name(DEFAULT_URL, 'bge-m3', ChunkingConfig(), BGE_M3)
        ensure_collection(self.store, self.name, 1024)
        self.embeddings = FakeEmbeddings(1024)
        for document in self.documents:
            index_chunks(list(chunk_article(document, CharacterTokenizer(), ChunkingConfig(), BGE_M3)),
                         document.text_hash, self.store, self.name, self.embeddings, profile=BGE_M3)
        self.embeddings.calls.clear()
        self.dense = [SearchResult(d.id, d.title, d.url, d.text, 1 - d.id / 10) for d in self.documents]

    def make_service(self, wait_ms: float = 100.0) -> AutoService:
        service = AutoService(self.store, self.name, CharacterTokenizer(), self.embeddings,
                              TantivyService(self.index), router=RouterConfig(lexical_wait_ms=wait_ms))
        self.addCleanup(service.close)
        return service

    def test_api_decision_and_final_snippets_use_one_embedding(self) -> None:
        service = self.make_service()
        with patch('habr_search.retrieval.auto.search_articles', return_value=self.dense), \
                patch.object(service.local.backend, 'render_candidates', wraps=service.local.backend.render_candidates) as render, \
                self.assertLogs('habr_search.auto', level='INFO') as logs, \
                TestClient(create_app(lambda: service)) as client:
            response = client.post('/api/search', json={'query': 'ExecuteExcel4Macro', 'top_k': 1})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data['results'][0]['article_id'], 2)
            self.assertEqual(data['diagnostics']['decision']['engine'], 'hybrid')
            self.assertEqual(data['diagnostics']['lexical_status'], 'ready')
            self.assertEqual(len(render.call_args.args[1]), 1)
            self.assertEqual(self.embeddings.calls, [['ExecuteExcel4Macro']])
            self.assertIn('route=hybrid', logs.output[-1])
            # даже техническое обозначение запроса не попадает в диагностический журнал
            self.assertNotIn('ExecuteExcel4Macro', '\n'.join(logs.output))
            before = len(self.embeddings.calls)
            self.assertEqual(client.post('/api/search', json={'query': 'a' * 81}).status_code, 400)
            self.assertEqual(len(self.embeddings.calls), before)

    def test_local_branch_starts_while_embedding_is_running(self) -> None:
        service = self.make_service()
        started, embedding_started = Event(), Event()
        select = service.local.backend.select_candidates
        embed = self.embeddings.embed

        def local(query: str, limit: int) -> list[LexicalCandidate]:
            started.set()
            self.assertTrue(embedding_started.wait(2))
            return select(query, limit)

        def remote(texts: list[str]) -> list[list[float]]:
            embedding_started.set()
            self.assertTrue(started.wait(2))
            return embed(texts)

        with patch.object(service.local.backend, 'select_candidates', side_effect=local), \
                patch.object(self.embeddings, 'embed', side_effect=remote), \
                patch('habr_search.retrieval.auto.search_articles', return_value=self.dense), \
                TestClient(create_app(lambda: service)) as client:
            response = client.post('/api/search', json={'query': 'документы'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['diagnostics']['decision']['engine'], 'bge-m3')
            self.assertEqual([r['article_id'] for r in response.json()['results']], [1, 2, 3])
            self.assertEqual(len(self.embeddings.calls), 1)

    def test_timeout_does_not_queue_queries_or_return_stale_results(self) -> None:
        service = self.make_service(wait_ms=0)
        started, release = Event(), Event()
        self.addCleanup(release.set)
        select = service.local.backend.select_candidates
        embed = self.embeddings.embed

        def slow(query: str, limit: int) -> list[LexicalCandidate]:
            started.set()
            self.assertTrue(release.wait(3))
            return select(query, limit)

        def remote(texts: list[str]) -> list[list[float]]:
            self.assertTrue(started.wait(2))
            return embed(texts)

        with patch.object(service.local.backend, 'select_candidates', side_effect=slow) as local, \
                patch.object(self.embeddings, 'embed', side_effect=remote), \
                patch('habr_search.retrieval.auto.search_articles', return_value=self.dense):
            # клиент закрывается только после освобождения заблокированного чтения
            with TestClient(create_app(lambda: service)) as client:
                try:
                    first = client.post('/api/search', json={'query': 'ExecuteExcel4Macro'}).json()
                    self.assertEqual(first['diagnostics']['lexical_status'], 'timeout')
                    second = client.post('/api/search', json={'query': 'телефония'}).json()
                    self.assertEqual(second['diagnostics']['lexical_status'], 'busy')
                    self.assertEqual(local.call_count, 1)
                    self.assertEqual([r['article_id'] for r in second['results']], [1, 2, 3])
                    self.assertEqual(len(self.embeddings.calls), 2)
                finally:
                    release.set()

    def test_lexical_errors_degrade_but_model_errors_are_not_hidden(self) -> None:
        service = self.make_service()
        with patch.object(service.local.backend, 'select_candidates', side_effect=ValueError('bad index')), \
                patch('habr_search.retrieval.auto.search_articles', return_value=self.dense), \
                self.assertLogs('habr_search.auto', level='WARNING'), \
                TestClient(create_app(lambda: service)) as client:
            data = client.post('/api/search', json={'query': 'ExecuteExcel4Macro'}).json()
            self.assertEqual(data['diagnostics']['decision']['reason'], 'lexical_error')
            self.assertEqual([r['article_id'] for r in data['results']], [1, 2, 3])
            with patch.object(self.embeddings, 'embed', side_effect=EmbeddingAPIError('model unavailable')):
                self.assertEqual(client.post('/api/search', json={'query': 'документы'}).status_code, 502)

    def test_snippet_error_falls_back_to_original_model_order(self) -> None:
        service = self.make_service()
        render = service.local.backend.render_candidates

        def broken(query: str, candidates: list[LexicalCandidate]) -> list[LexicalResult]:
            if candidates:
                raise RuntimeError('snippet unavailable')
            return render(query, candidates)

        with patch.object(service.local.backend, 'render_candidates', side_effect=broken), \
                patch('habr_search.retrieval.auto.search_articles', return_value=self.dense), \
                self.assertLogs('habr_search.auto', level='WARNING'), \
                TestClient(create_app(lambda: service)) as client:
            data = client.post('/api/search', json={'query': 'ExecuteExcel4Macro', 'top_k': 1}).json()
            self.assertEqual(data['results'][0]['article_id'], 1)
            self.assertEqual(data['diagnostics']['decision']['reason'], 'lexical_snippet_error')

    def test_cli_and_offline_replay_never_make_unexpected_paid_calls(self) -> None:
        self.store.close()
        # cli проверяется на отдельном временном векторном хранилище
        store_path = self.root / 'qdrant'
        with closing(QdrantClient(path=str(store_path))) as store:
            ensure_collection(store, self.name, 1024)
            for document in self.documents:
                index_chunks(list(chunk_article(document, CharacterTokenizer(), ChunkingConfig(), BGE_M3)),
                             document.text_hash, store, self.name, self.embeddings, profile=BGE_M3)
        self.embeddings.calls.clear()

        def serve(app: FastAPI, **_options: object) -> None:
            with TestClient(app) as client:
                self.assertEqual(client.post('/api/search', json={'query': 'ExecuteExcel4Macro'}).status_code, 200)

        with patch('sys.argv', ['web', '--engine', 'auto', '--qdrant-path', str(store_path),
                               '--tantivy-path', str(self.index)]), redirect_stdout(io.StringIO()), \
                patch('habr_search.web.app.read_api_key', return_value='test'), \
                patch('habr_search.web.app.load_tokenizer', return_value=CharacterTokenizer()), \
                patch('habr_search.web.app.EmbeddingsClient', return_value=self.embeddings), \
                patch('habr_search.web.app.uvicorn.run', side_effect=serve):
            self.assertEqual(web_main(), 0)
        self.assertEqual(self.embeddings.calls, [['ExecuteExcel4Macro']])
        local = TantivyService(self.index, spelling=False).backend
        context = ComparisonContext(self.name, 'revision', local.info.revision, 3, frozenset((1, 2, 3)),
                                    (EvaluationQuery('macro', 'exact', 'ExecuteExcel4Macro', (2,)),), HybridConfig())
        dense = tuple(RankedDocument(item.article_id, item.score) for item in self.dense)
        lexical = (RankedDocument(2, 10),)
        hybrid = tuple(RankedDocument(item.article_id, item.score)
                       for item in fuse_candidates([1, 2, 3], [2], context.config))
        record = ComparisonRecord('macro', '2026-11-17T00:00:00+00:00', dense, lexical, hybrid, 1, 1, 2)
        replayed = replay_auto(context, {'macro': record}, local.article_titles())
        self.assertEqual(replayed['macro'].ranking[0].article_id, 2)
        run = self.root / 'run'
        context.save(run)
        with (run / 'rankings.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            writer.writerow(record.row())
        output = self.root / 'report'
        args = ['evaluate', '--from-run', str(run), '--tantivy-path', str(self.index), '--output', str(output)]
        with patch('sys.argv', args), redirect_stdout(io.StringIO()), \
                patch('habr_search.embeddings.EmbeddingsClient', side_effect=AssertionError('offline')):
            self.assertEqual(evaluate_main(), 0)
        self.assertTrue((output / 'summary.csv').is_file())
        metadata = json.loads((output / 'metadata.json').read_text())
        self.assertEqual(metadata['mws_calls'], 0)
        self.assertEqual(metadata['timing_scope'], 'not_measured_offline')
        before = (output / 'summary.csv').read_bytes()
        with patch('sys.argv', args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            evaluate_main()
        self.assertEqual(before, (output / 'summary.csv').read_bytes())


if __name__ == '__main__':
    unittest.main()
