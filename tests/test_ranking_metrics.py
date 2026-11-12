"""сверяем метрики с ручными расчётами, включая несколько ответов, равные оценки и пустую выдачу."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from habr_search.embeddings import DEFAULT_URL
from habr_search.evaluation.compare import main
from habr_search.evaluation.comparison_data import (
    COLUMNS, ComparisonContext, ComparisonRecord, load_context, load_judgments, load_records,
)
from habr_search.evaluation.comparison_metrics import create_metrics
from habr_search.evaluation.evaluate import EvaluationQuery, load_queries
from habr_search.evaluation.ranking_metrics import RankedDocument, evaluate_ranking
from habr_search.ingestion.chunking import ChunkingConfig
from habr_search.ingestion.index import collection_name, ensure_collection
from habr_search.model_profiles import BGE_M3
from habr_search.paths import PROJECT_ROOT
from habr_search.evaluation.prepare_judgments import prepare_pool
from habr_search.evaluation.evaluate_hybrid import HybridContext, HybridEvaluation, REPORT_COLUMNS
from habr_search.evaluation.evaluate import query_set_hash
from habr_search.retrieval.hybrid import HybridConfig, fuse_candidates, fuse_rankings
from habr_search.retrieval.tantivy_index import TantivySearch, build_index
from habr_search.vector_store import QdrantConfig
from tests.test_chunking import CharacterTokenizer
from tests.test_evaluate import FakeEmbeddings


def ranking(*ids: int) -> tuple[RankedDocument, ...]:
    """каждому следующему документу даём меньшую оценку, без вызова модели."""
    return tuple(RankedDocument(identifier, 1 - index / 10) for index, identifier in enumerate(ids))


class MetricFormulaTests(unittest.TestCase):
    def test_multiple_relevant_documents_and_graded_ndcg(self) -> None:
        score = evaluate_ranking(ranking(2, 1, 4, 3, 5), {1: 3, 3: 1, 5: 2}, frozenset(range(1, 6)), 2, [1, 3, 5])
        self.assertAlmostEqual(score.values['RR'], .5)
        self.assertAlmostEqual(score.values['Precision'], .5)
        self.assertAlmostEqual(score.values['Recall'], 1 / 3)
        self.assertAlmostEqual(score.values['F1'], .4)
        self.assertAlmostEqual(score.values['AP'], (.5 + .5 + .6) / 3)
        self.assertAlmostEqual(score.values['AP@3'], .5 / 3)
        self.assertAlmostEqual(score.values['Precision@3'], 1 / 3)
        self.assertAlmostEqual(score.values['Recall@3'], 1 / 3)
        self.assertEqual(score.values['RR@1'], 0)
        self.assertAlmostEqual(score.values['nDCG@3'], (7 / math.log2(3)) / (7 + 3 / math.log2(3) + .5))
        self.assertAlmostEqual(score.values['PR-AUC'], 73 / 180)
        self.assertNotAlmostEqual(score.values['AP'], score.values['PR-AUC'])
        self.assertEqual(score.curve[-1].recall, 1)
        self.assertEqual(score.curve[-1].retrieved, 5)

    def test_short_list_does_not_change_precision_at_k_denominator(self) -> None:
        score = evaluate_ranking(ranking(1), {1: 1, 2: 1}, frozenset({1, 2, 3}), 5, [5], missing_score=0)
        self.assertEqual(score.values['Precision'], 1)
        self.assertEqual(score.values['Precision@5'], .2)
        self.assertEqual(score.values['Recall@5'], .5)
        self.assertEqual(score.values['AP'], .5)
        # недостающий релевантный документ учитывается в полном знаменателе ap@k
        self.assertEqual(score.values['AP@5'], .5)

    def test_score_ties_are_one_pr_threshold(self) -> None:
        docs = (RankedDocument(1, .5), RankedDocument(2, .5))
        score = evaluate_ranking(docs, {1: 1, 2: 0}, frozenset({1, 2}), cutoffs=[2])
        self.assertEqual(len(score.curve), 2)
        self.assertEqual(score.values['PR-AUC'], .75)
        self.assertEqual(score.values['AP'], 1)
        reversed_score = evaluate_ranking(docs[::-1], {1: 1}, frozenset({1, 2}), cutoffs=[2])
        self.assertEqual(reversed_score.values['PR-AUC'], .75)
        self.assertEqual(reversed_score.values['AP'], .5)

    def test_empty_lexical_search_is_not_a_perfect_retrieval(self) -> None:
        score = evaluate_ranking((), {1: 1}, frozenset({1, 2, 3}), cutoffs=[5], missing_score=0)
        for name in ('RR', 'Precision', 'Recall', 'F1', 'AP', 'Precision@5', 'Recall@5', 'nDCG@5'):
            self.assertEqual(score.values[name], 0)
        # у постоянной оценки трапецеидальная pr-auc имеет ненулевой базовый уровень
        self.assertAlmostEqual(score.values['PR-AUC'], 2 / 3)

    def test_negative_queries_have_no_rank_metrics_and_unknowns_are_counted(self) -> None:
        score = evaluate_ranking(ranking(1, 2), {}, frozenset({1, 2}), cutoffs=[1])
        self.assertEqual(score.values, {})
        self.assertEqual(score.unjudged_returned, 2)
        self.assertEqual(score.relevant_total, 0)

    def test_invalid_rankings_labels_and_missing_dense_scores_are_rejected(self) -> None:
        invalid = [ranking(1, 1), ranking(3), (RankedDocument(1, math.nan),),
                   (RankedDocument(1, .1), RankedDocument(2, .2))]
        for docs in invalid:
            with self.subTest(docs=docs), self.assertRaises(ValueError):
                evaluate_ranking(docs, {1: 1}, frozenset({1, 2}), cutoffs=[1])
        for grades in ({1: 4}, {3: 1}):
            with self.assertRaises(ValueError):
                evaluate_ranking(ranking(1, 2), grades, frozenset({1, 2}), cutoffs=[1])
        with self.assertRaisesRegex(ValueError, 'pr-auc'):
            evaluate_ranking(ranking(1), {1: 1}, frozenset({1, 2}), cutoffs=[1])
        with self.assertRaises(ValueError):
            evaluate_ranking(ranking(1, 2), {1: 1}, frozenset({1, 2}), cutoffs=[1, 1])

    def test_full_fusion_preserves_the_web_prefix(self) -> None:
        config = HybridConfig(2)
        full = fuse_candidates([1, 2], [3, 2], config)
        self.assertEqual(full[:2], fuse_rankings([1, 2], [3, 2], 2, config))
        self.assertEqual(len(full), 3)


class SavedComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.context = ComparisonContext('test', 'q-v1', 't-v1', 3, frozenset({1, 2, 3}),
                                         (EvaluationQuery('one', 'exact', 'отчёт', (1,)),
                                          EvaluationQuery('none', 'no_answer', 'нет ответа', ())), HybridConfig())
        self.run_folder = self.root / 'run'
        self.context.save(self.run_folder)
        dense, lexical = ranking(2, 1, 3), ranking(1, 2)
        hybrid = tuple(RankedDocument(d.article_id, d.score) for d in fuse_candidates([2, 1, 3], [1, 2]))
        self.records = {case.query_id: ComparisonRecord(case.query_id, '2026-11-12T12:00:00+00:00',
                                                       dense, lexical, hybrid, 100, 20, 125)
                        for case in self.context.cases}
        self.write_records()

    def write_records(self) -> None:
        with (self.run_folder / 'rankings.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(record.row() for record in self.records.values())

    def test_context_and_full_scores_round_trip(self) -> None:
        self.assertEqual(load_context(self.run_folder), self.context)
        self.assertEqual(load_records(self.run_folder, self.context), self.records)

    def test_expanding_labels_and_removing_wrong_seed_without_changing_run(self) -> None:
        labels = self.root / 'qrels.csv'
        labels.write_text('query_id,article_id,relevance,comment\none,2,3,ещё одна подходящая статья\nnone,1,0,проверено\n')
        judgments = load_judgments(self.context, labels)
        self.assertEqual(judgments['one'], {1: 1, 2: 3})
        self.assertEqual(judgments['none'], {1: 0})
        before = (self.run_folder / 'rankings.csv').read_bytes()
        destination = self.root / 'metrics'
        with redirect_stdout(io.StringIO()):
            create_metrics(self.context, self.records, destination, 5, [1, 5], labels)
        with (destination / 'summary.csv').open() as source:
            rows = list(csv.DictReader(source))
        dense = next(row for row in rows if row['engine'] == 'bge-m3' and row['category'] == 'all')
        self.assertEqual(float(dense['MRR']), 1)
        self.assertEqual(float(dense['Recall@5']), 1)
        self.assertEqual(float(dense['Precision@5']), .4)
        self.assertEqual(dense['positive_queries'], '1')
        self.assertEqual(dense['negative_queries'], '1')
        self.assertEqual((self.run_folder / 'rankings.csv').read_bytes(), before)
        with (destination / 'judgments.csv').open() as source:
            pool = list(csv.DictReader(source))
        self.assertTrue(any(row['article_id'] == '3' and row['relevance'] == '' for row in pool))

    def test_offline_cli_never_connects_to_resources_or_model(self) -> None:
        with patch('sys.argv', ['compare', '--from-run', str(self.run_folder), '--report-output', str(self.root / 'report')]), \
                patch('habr_search.evaluation.compare.connect_store', side_effect=AssertionError('индекс не нужен')), \
                patch('habr_search.evaluation.compare.read_api_key', side_effect=AssertionError('ключ не нужен')), \
                patch('habr_search.evaluation.compare.EmbeddingsClient', side_effect=AssertionError('mws не нужен')), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)

    def test_corrupted_cache_and_invalid_qrels_are_rejected(self) -> None:
        for text in ('one,1,4', 'other,1,1', 'one,99,1', 'one,1,1\none,1,2', 'one,1,0', 'none,2,1'):
            labels = self.root / 'labels.csv'
            labels.write_text('query_id,article_id,relevance\n' + text + '\n')
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_judgments(self.context, labels)
        path = self.run_folder / 'rankings.csv'
        path.write_text(path.read_text().replace('2;1;3', '2;1', 1))
        with self.assertRaises(ValueError):
            load_records(self.run_folder, self.context)

    def test_incomplete_run_cannot_be_reported_as_a_complete_comparison(self) -> None:
        with self.assertRaises(ValueError):
            create_metrics(self.context, {'one': self.records['one']}, self.root / 'incomplete', 5, [5])
        self.assertFalse((self.root / 'incomplete').exists())


class JudgmentPoolTests(unittest.TestCase):
    def test_pool_contains_candidates_from_both_sources_without_fake_labels(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            queries = root / 'queries.csv'
            queries.write_text('query_id,category,query,relevant_article_ids\nq,exact,отчёты,1\n')
            cases = load_queries(queries)
            context = HybridContext('test', 'q-v1', 't-v1', 3, query_set_hash(cases), 2, HybridConfig())
            dense, lexical = (2, 1), (3, 1)
            combined = tuple(d.article_id for d in fuse_rankings(dense, lexical, 2))
            record = HybridEvaluation(cases[0], dense, lexical, combined, 120, 100, 20, '2026-11-12T12:00:00+00:00')
            report = root / 'old-report.csv'
            with report.open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(REPORT_COLUMNS)
                writer.writerow(record.row(context))
            corpus = root / 'corpus.sqlite3'
            with sqlite3.connect(corpus) as db:
                db.execute('CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
                for i in (1, 2, 3):
                    db.execute('INSERT INTO articles VALUES (?,?)', (i, json.dumps({
                        'id': i, 'url': f'https://habr.com/ru/articles/{i}/', 'title': f'отчёт {i}', 'text': f'текст {i}' * 30})))
            output = root / 'judgments.csv'
            self.assertEqual(prepare_pool(queries, report, corpus, output, 2), 3)
            with output.open() as source:
                rows = {int(row['article_id']): row for row in csv.DictReader(source)}
            self.assertEqual(rows[1]['relevance'], '1')
            self.assertEqual(rows[2]['relevance'], '')
            self.assertEqual(rows[3]['relevance'], '')
            self.assertEqual(rows[1]['label_source'], 'seed_needs_review')
            self.assertEqual(rows[2]['bge_m3_rank'], '1')
            self.assertEqual(rows[3]['tantivy_rank'], '1')
            with self.assertRaises(ValueError):
                prepare_pool(queries, report, corpus, output, 2)

    def test_multirelevant_diagnostic_has_multiple_seed_answers_and_explicit_scope(self) -> None:
        cases = load_queries(PROJECT_ROOT / 'evaluation/queries_multirelevant.csv')
        self.assertEqual(len(cases), 6)
        self.assertTrue(all(2 <= len(case.relevant_ids) <= 4 for case in cases))
        context = ComparisonContext('test', 'v1', 'v2', 7,
                                     frozenset(i for case in cases for i in case.relevant_ids), tuple(cases), HybridConfig())
        labels = load_judgments(context, PROJECT_ROOT / 'evaluation/qrels_multirelevant.csv')
        for case in cases:
            self.assertEqual({i for i, grade in labels[case.query_id].items() if grade > 0}, set(case.relevant_ids))
        with (PROJECT_ROOT / 'evaluation/multirelevant_manifest.json').open() as stream:
            metadata = json.load(stream)
        self.assertFalse(metadata['independent_holdout'])
        self.assertEqual(metadata['annotation_status'], 'content_checked_seed_labels_not_exhaustive')


class ComparisonPipelineTests(unittest.TestCase):
    def test_one_embedding_per_query_and_resume_after_failure(self) -> None:
        with TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            root = Path(folder)
            corpus = root / 'corpus.sqlite3'
            with sqlite3.connect(corpus) as db:
                db.execute('CREATE TABLE articles (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
                for identifier, text in ((1, 'автоматизация отчётов Word'), (2, 'отчёты Word Excel'), (3, 'процессы Linux')):
                    db.execute('INSERT INTO articles VALUES (?, ?)', (identifier, json.dumps({
                        'id': identifier, 'url': f'https://habr.com/ru/articles/{identifier}/',
                        'title': text, 'text': text * 20})))
            index = root / 'tantivy'
            build_index(corpus, index)
            lexical = TantivySearch(index)
            self.assertEqual({i for i, _ in lexical.ranked_matches('Word')}, {1, 2})
            self.assertEqual(lexical.ranked_matches('несуществующееслово'), ())
            name = collection_name(DEFAULT_URL, 'bge-m3', ChunkingConfig(), BGE_M3)

            def store(_config: QdrantConfig) -> QdrantClient:
                client = QdrantClient(':memory:')
                ensure_collection(client, name, 1024)
                client.upsert(name, [models.PointStruct(id=i, vector=[1.0] + [0.0] * 1023,
                                                        payload={'article_id': i}) for i in (1, 2, 3)])
                return client

            queries = root / 'queries.csv'
            queries.write_text('query_id,category,query,relevant_article_ids\nword,exact,Word,1;2\nlinux,exact,Linux,3\n')
            common = ['--queries', str(queries), '--tantivy-path', str(index), '--qdrant-url', 'http://127.0.0.1:6333']
            run = root / 'run'
            first = FakeEmbeddings(1024, fail_on_call=2)
            second = FakeEmbeddings(1024)
            with patch('habr_search.evaluation.compare.connect_store', side_effect=store), \
                    patch('habr_search.evaluation.compare.load_tokenizer', return_value=CharacterTokenizer()), \
                    patch('habr_search.evaluation.compare.read_api_key', return_value='test'), \
                    patch('habr_search.evaluation.compare.EmbeddingsClient', side_effect=[first, second]), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with patch('sys.argv', ['compare', *common, '--output', str(run)]), self.assertRaises(SystemExit):
                    main()
                self.assertEqual(len(load_records(run, load_context(run))), 1)
                with patch('sys.argv', ['compare', *common, '--resume', str(run)]):
                    self.assertEqual(main(), 0)
            self.assertEqual(len(first.calls), 2)
            self.assertEqual(len(second.calls), 1)
            self.assertTrue(first.closed and second.closed)
            records = load_records(run, load_context(run))
            self.assertEqual(set(records), {'word', 'linux'})
            self.assertEqual(len(records['word'].dense), 3)
            self.assertEqual(len(records['word'].lexical), 2)


if __name__ == '__main__':
    unittest.main()
