"""проверяем порядок, состав пакета и обработку ошибок mws."""
import unittest
from unittest.mock import MagicMock

import requests

from habr_search.retrieval.hybrid import HybridResult
from habr_search.retrieval.rerank import Reranker, RerankError, parse_order


class RerankTests(unittest.TestCase):
    def test_batch_and_tail(self) -> None:
        session = MagicMock(spec=requests.Session)
        session.headers = {}
        reply = session.post.return_value.__enter__.return_value
        reply.status_code = 200
        reply.json.return_value = {'results': [{'index': 1, 'relevance_score': 1}, {'index': 0, 'relevance_score': 0}]}
        client = Reranker(session, 'test', 'reranker', 2)
        items = [HybridResult(i, str(i), '', 'фрагмент', 0.1) for i in range(1, 4)]
        self.assertEqual(client.rank('запрос', items), [items[1], items[0], items[2]])
        session.post.assert_called_once()
        self.assertEqual(session.post.call_args.kwargs['json']['documents'], ['1\nфрагмент', '2\nфрагмент'])
        session.post.side_effect = requests.Timeout()
        with self.assertRaises(RerankError):
            client.rank('запрос', items)

    def test_invalid_and_tied_scores(self) -> None:
        row = {'index': 0, 'relevance_score': 1}
        for rows in ([row], [row, row], [row, dict(index=2, relevance_score=0)],
                     [row, dict(index=1, relevance_score=float('nan'))]):
            with self.assertRaises(RerankError):
                parse_order({'results': rows}, 2)
        self.assertEqual(parse_order({'results': [dict(index=1, relevance_score=1), row]}, 2), [0, 1])
