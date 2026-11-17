"""проверяем признаки, независимые темы и настоящий lightgbm без платной модели."""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import re
import sqlite3
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from habr_search.embeddings import DEFAULT_URL
from habr_search.evaluation.comparison_data import COLUMNS, ComparisonContext, ComparisonRecord, load_judgments
from habr_search.evaluation.evaluate import EvaluationQuery
from habr_search.evaluation.ltr_data import training_groups
from habr_search.evaluation.train_ltr import main as train_main
from habr_search.index_version import index_revision
from habr_search.ingestion.chunking import ChunkingConfig, chunk_article
from habr_search.ingestion.index import collection_name, ensure_collection, index_chunks
from habr_search.model_profiles import BGE_M3
from habr_search.retrieval.hybrid import HybridConfig, fuse_candidates
from habr_search.retrieval.ltr import LearnedRanker, require_lightgbm
from habr_search.retrieval.ltr_features import FEATURE_NAMES, FeatureCatalog, FeatureDocument, RankSignal, build_features, literal_match
from habr_search.evaluation.ranking_metrics import RankedDocument
from habr_search.retrieval.router import RouterConfig
from habr_search.retrieval.search import SearchResult
from habr_search.retrieval.tantivy_index import TantivySearch, build_index
from habr_search.web.app import AutoService, TantivyService, create_app, main as web_main
from tests.test_chunking import CharacterTokenizer, article
from tests.test_evaluate import FakeEmbeddings


class FeatureTests(unittest.TestCase):
    def test_literal_identifier_matching_preserves_regex_word_boundaries(self) -> None:
        for text in ('python310 python3', 'foo_foo', 'foo-bar', 'afoo foo', 'c++ code', 'яfoo fooя',
                     'Application.OnTime()', 'foo.bar and foo', 'текст точная фраза текст', 'Ⅳfoo fooⅣ'):
            for needle in ('python3', 'foo', 'c++', 'application.ontime', 'foo.bar', 'точная фраза'):
                expected = bool(re.search(r'(?<!\w)' + re.escape(needle) + r'(?!\w)', text.casefold()))
                self.assertEqual(literal_match(text.casefold(), needle), expected, (text, needle))

    def test_missing_source_scores_and_body_matches_are_explicit(self) -> None:
        catalog = FeatureCatalog({1: FeatureDocument.from_text('другая статья', 'описание таблиц'),
                                  2: FeatureDocument.from_text('общий заголовок', 'Пример ERR_CONNECTION_RESET в тексте')})
        found = build_features('ERR_CONNECTION_RESET', [RankSignal(1, 0.8)], [RankSignal(2, 10)], catalog, 20)
        values = dict(zip(FEATURE_NAMES, found[1].values))
        self.assertEqual(values['dense_present'], 0)
        self.assertEqual(values['dense_score'], 0)
        self.assertEqual(values['lexical_present'], 1)
        self.assertEqual(values['body_identifier_coverage'], 1)
        self.assertEqual(values['title_identifier_coverage'], 0)
        # id и метки не входят в числовую матрицу
        self.assertEqual(len(found[1].values), len(FEATURE_NAMES))
        groups = training_groups({'q': found}, {'q': {1: 0, 2: 3}}, {'q': 'topic'})
        self.assertEqual(groups[0].grades, (0, 3))
        self.assertEqual(training_groups({'q': found}, {'q': {2: 3}}, {'q': 'topic'}), [])
        with self.assertRaises(ValueError):
            build_features('x', [RankSignal(1, 0.2), RankSignal(1, 0.3)], [], catalog, 20)

    def test_optional_dependency_error_explains_installation(self) -> None:
        with patch('habr_search.retrieval.ltr.importlib.import_module', side_effect=ImportError), \
                self.assertRaisesRegex(RuntimeError, 'requirements-ltr'):
            require_lightgbm()


@unittest.skipUnless(importlib.util.find_spec('lightgbm') is not None, 'нужны requirements-ltr.txt')
class LearnedRankerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = self.root / 'tantivy'
        self.output = self.root / 'model'
        self.source = self.root / 'corpus.sqlite3'
        self.documents = [article(f'описание {name} и пример работы', name, identifier)
                          for identifier, name in enumerate(('AlphaEngine', 'BetaEngine', 'GammaEngine', 'UnknownEngine'), 1)]
        with sqlite3.connect(self.source) as connection:
            connection.execute('CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
            for document in self.documents:
                connection.execute('INSERT INTO articles VALUES (?,?)', (document.id, json.dumps({
                    'id': document.id, 'title': document.title, 'url': document.url, 'text': document.text})))
        build_index(self.source, self.snapshot)
        self.store = QdrantClient(':memory:')
        self.addCleanup(self.store.close)
        self.name = collection_name(DEFAULT_URL, 'bge-m3', ChunkingConfig(), BGE_M3)
        ensure_collection(self.store, self.name, 1024)
        self.embeddings = FakeEmbeddings(1024)
        for document in self.documents:
            index_chunks(list(chunk_article(document, CharacterTokenizer(), ChunkingConfig(), BGE_M3)),
                         document.text_hash, self.store, self.name, self.embeddings, profile=BGE_M3)
        self.embeddings.calls.clear()
        self.local = TantivySearch(self.snapshot)
        self.cases = tuple(EvaluationQuery(f'q{identifier}_{variant}', 'exact' if variant == 0 else 'paraphrase',
                           document.title if variant == 0 else 'как работает ' + document.title, (identifier, 4))
                           for identifier, document in enumerate(self.documents[:3], 1) for variant in range(2))
        self.context = ComparisonContext(self.name, index_revision(self.store, self.name), self.local.info.revision,
                                         4, frozenset((1, 2, 3, 4)), self.cases, HybridConfig())
        self.run_folder = self.root / 'run'
        self.context.save(self.run_folder)
        with (self.run_folder / 'rankings.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            for case in self.cases:
                dense = tuple(RankedDocument(i, 1 - i / 10) for i in (1, 2, 3, 4))
                order = [case.relevant_ids[0], *(i for i in (1, 2, 3, 4) if i != case.relevant_ids[0])]
                lexical = tuple(RankedDocument(i, float(10 - rank)) for rank, i in enumerate(order))
                fused = tuple(RankedDocument(i.article_id, i.score) for i in fuse_candidates([1, 2, 3, 4], order))
                writer.writerow(ComparisonRecord(case.query_id, '2026-11-17T00:00:00+00:00', dense, lexical, fused, 1, 1, 2).row())
        self.qrels = self.root / 'qrels.csv'
        self.topics = self.root / 'topics.csv'
        with self.qrels.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['query_id', 'article_id', 'relevance'])
            for case in self.cases:
                writer.writerows([case.query_id, identifier, 3 if identifier in case.relevant_ids else 0]
                                 for identifier in (1, 2, 3))
        with self.topics.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['query_id', 'topic'])
            writer.writerows([case.query_id, case.query_id.split('_')[0]] for case in self.cases)

    def train(self) -> LearnedRanker:
        args = ['train', '--from-run', str(self.run_folder), '--qrels', str(self.qrels), '--groups', str(self.topics),
                '--tantivy-path', str(self.snapshot), '--output', str(self.output), '--trees', '20']
        with patch('sys.argv', args), redirect_stdout(io.StringIO()), \
                patch('habr_search.embeddings.EmbeddingsClient', side_effect=AssertionError('no mws')):
            self.assertEqual(train_main(), 0)
        return LearnedRanker(self.output)

    def test_training_uses_only_explicit_grades_and_disjoint_topic_folds(self) -> None:
        self.assertNotIn(4, load_judgments(self.context, self.qrels, include_seeds=False)[self.cases[0].query_id])
        model = self.train()
        metadata = model.metadata
        self.assertEqual(metadata['training_pairs'], 18)
        self.assertEqual(metadata['mws_calls'], 0)
        folds = json.loads((self.output / 'metadata.json').read_text())['folds']
        for fold in folds:
            train_topics = {q.split('_')[0] for q in fold['train_query_ids']}
            held_topics = {q.split('_')[0] for q in fold['validation_query_ids']}
            self.assertFalse(train_topics & held_topics)
        catalog = self.local.feature_catalog()
        signals = [RankSignal(i, 1 - i / 10) for i in (1, 2, 3, 4)]
        before = model.rank('BetaEngine', signals, [RankSignal(2, 10)], catalog)
        after = LearnedRanker(self.output).rank('BetaEngine', signals, [RankSignal(2, 10)], catalog)
        self.assertEqual(before, after)
        self.assertEqual(len({item.article_id for item in after}), 4)
        with self.assertRaisesRegex(ValueError, 'индексы'):
            model.validate_context(self.name, 'wrong-version', self.local.info.revision, 50)

    def test_api_has_one_embedding_and_falls_back_on_ranker_failure(self) -> None:
        model = self.train()
        service = AutoService(self.store, self.name, CharacterTokenizer(), self.embeddings,
                              TantivyService(self.snapshot), HybridConfig(50), RouterConfig(lexical_wait_ms=100), model)
        self.addCleanup(service.close)
        dense = [SearchResult(d.id, d.title, d.url, d.text, 1 - d.id / 10) for d in self.documents]
        with patch('habr_search.retrieval.auto.search_articles', return_value=dense), \
                TestClient(create_app(lambda: service)) as client:
            result = client.post('/api/search', json={'query': 'BetaEngine', 'top_k': 3})
            self.assertEqual(result.status_code, 200)
            data = result.json()
            self.assertEqual(data['diagnostics']['decision']['engine'], 'ltr')
            self.assertEqual(data['diagnostics']['ranker_model_sha256'], model.model_hash)
            self.assertGreaterEqual(data['diagnostics']['ranker_ms'], 0)
            self.assertEqual(len(data['results']), 3)
            self.assertEqual(self.embeddings.calls, [['BetaEngine']])
            with patch.object(model, 'rank', side_effect=RuntimeError('bad predictor')), self.assertLogs('habr_search.auto'):
                failed = client.post('/api/search', json={'query': 'BetaEngine', 'top_k': 3}).json()
            self.assertEqual(failed['diagnostics']['decision']['reason'], 'ranker_error')
            self.assertEqual([item['article_id'] for item in failed['results']], [1, 2, 3])
            self.assertEqual(len(self.embeddings.calls), 2)

    def test_corrupt_artifact_and_missing_cli_model_fail_before_payment(self) -> None:
        self.train()
        with (self.output / 'model.txt').open('a') as stream:
            stream.write('corrupted')
        with self.assertRaisesRegex(ValueError, 'сумма'):
            LearnedRanker(self.output)
        with patch('sys.argv', ['web', '--engine', 'ltr']), redirect_stderr(io.StringIO()), \
                patch('habr_search.web.app.read_api_key', side_effect=AssertionError('no key')), self.assertRaises(SystemExit):
            web_main()

    def test_cli_loads_ranker_depth_and_starts_existing_api(self) -> None:
        model = self.train()

        def serve(app: object, **_options: object) -> None:
            from fastapi import FastAPI
            if not isinstance(app, FastAPI):
                raise AssertionError('ожидался fastapi')
            with TestClient(app) as client:
                data = client.post('/api/search', json={'query': 'BetaEngine'}).json()
                self.assertEqual(data['diagnostics']['decision']['engine'], 'ltr')
                self.assertEqual(data['diagnostics']['ranker_model_sha256'], model.model_hash)

        with patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()), \
                patch('sys.argv', ['web', '--engine', 'ltr', '--ranker-model', str(self.output),
                                   '--tantivy-path', str(self.snapshot), '--qdrant-url', 'http://127.0.0.1:6333',
                                   '--auto-lexical-wait-ms', '100']), \
                patch('habr_search.web.app.connect_store', return_value=self.store), \
                patch('habr_search.web.app.read_api_key', return_value='test'), \
                patch('habr_search.web.app.EmbeddingsClient', return_value=self.embeddings), \
                patch('habr_search.web.app.load_tokenizer', return_value=CharacterTokenizer()), \
                patch('habr_search.web.app.uvicorn.run', side_effect=serve):
            self.assertEqual(web_main(), 0)
        self.assertEqual(self.embeddings.calls, [['BetaEngine']])


if __name__ == '__main__':
    unittest.main()
