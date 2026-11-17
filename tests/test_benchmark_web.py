"""проверяем http-замеры, число попыток и остановку процессов без mws."""
from __future__ import annotations

import csv
import io
import json
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import cast
from unittest.mock import Mock, patch

import requests

from habr_search.evaluation.benchmark_web import ENGINES, collect_timings, main, stop_server
from habr_search.evaluation.evaluate import EvaluationQuery


class BenchmarkWebTests(unittest.TestCase):
    def server(self, fail_engine: str = '') -> tuple[str, list[str]]:
        attempts: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                engine = self.path.split('/')[1]
                attempts.append(engine)
                length = int(self.headers.get('Content-Length', '0'))
                request = json.loads(self.rfile.read(length))
                if engine == fail_engine:
                    self.send_response(503)
                    self.end_headers()
                    return
                body = json.dumps({'original_query': request['query'], 'used_query': request['query'],
                                   'corrected': False, 'elapsed_ms': 1.0, 'results': []}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close() -> None:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.addCleanup(close)
        return f'http://127.0.0.1:{server.server_port}', attempts

    def test_actual_http_requests_are_rotated_and_not_replayed(self) -> None:
        url, attempts = self.server()
        addresses = {engine: url + '/' + engine for engine in ENGINES}
        queries = [EvaluationQuery('first', 'exact', 'Application.OnTime', (1,)),
                   EvaluationQuery('second', 'quoted', '"точная фраза"', (2,))]
        stream = io.StringIO()
        with requests.Session() as session, redirect_stdout(io.StringIO()):
            session.trust_env = False
            times = collect_timings(session, addresses, queries, 2, 20, 3, stream)
        self.assertEqual(len(times), 12)
        self.assertEqual(attempts[:6], ['bge-m3', 'hybrid', 'auto', 'hybrid', 'auto', 'bge-m3'])
        self.assertTrue(all(time.http_ms > 0 for time in times))
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({(r['query_id'], r['engine'], r['repeat']) for r in rows}), 12)
        self.assertTrue(all(r['status'] == 'ok' for r in rows))

    def test_error_is_saved_and_never_automatically_retried(self) -> None:
        url, attempts = self.server(fail_engine='hybrid')
        stream = io.StringIO()
        with requests.Session() as session, redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            session.trust_env = False
            collect_timings(session, {engine: url + '/' + engine for engine in ENGINES},
                            [EvaluationQuery('q', 'exact', 'запрос', (1,))], 1, 20, 3, stream)
        self.assertEqual(attempts, ['bge-m3', 'hybrid'])
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(rows[-1]['status'], 'error')
        self.assertEqual(rows[-1]['error_type'], 'HTTPError')

    def test_only_owned_running_process_is_terminated(self) -> None:
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('test', 10), 0]
        stop_server(cast(subprocess.Popen[bytes], process))
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        process.poll.return_value = 0
        stop_server(cast(subprocess.Popen[bytes], process))
        process.terminate.assert_called_once()

    def test_existing_output_is_rejected_before_key_or_servers(self) -> None:
        with TemporaryDirectory() as directory, patch('sys.argv', ['benchmark', '--output', directory]), \
                patch('habr_search.evaluation.benchmark_web.read_api_key', side_effect=AssertionError('no key')), \
                patch('habr_search.evaluation.benchmark_web.subprocess.Popen', side_effect=AssertionError('no server')), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main()
        self.assertFalse(Path(directory).exists())


if __name__ == '__main__':
    unittest.main()
