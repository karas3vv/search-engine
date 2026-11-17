#!/usr/bin/env bash
# проверяем качество и реальную задержку auto одним воспроизводимым запуском
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x .venv/bin/python ]]; then
    python_bin="$PWD/.venv/bin/python"
else
    python_bin=python3
fi
if [[ $# -gt 1 ]]; then
    echo 'использование: ./check_auto.sh [новая папка отчёта]' >&2
    exit 1
fi
report_dir="${1:-data/evaluation/auto-check-$(date -u +%Y%m%d-%H%M%S)}"
if [[ -e "$report_dir" ]]; then
    echo 'папка отчёта уже существует; укажите новую, прежние результаты не перезаписываются' >&2
    exit 1
fi

# индексы проверяем до ввода ключа и платных запросов
docker compose up -d qdrant
"$python_bin" - <<'PY'
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener
client = build_opener(ProxyHandler({}))
for attempt in range(30):
    try:
        with client.open('http://127.0.0.1:6333/collections', timeout=1):
            break
    except (URLError, OSError):
        time.sleep(1)
else:
    raise SystemExit('qdrant не ответил; проверьте контейнер')
PY
"$python_bin" -m habr_search.evaluation.compare \
    --queries evaluation/queries_auto.csv --qrels evaluation/qrels_auto.csv \
    --qdrant-url http://127.0.0.1:6333 --check-only

echo 'запуск: 24 запроса MWS для качества + 18 для скорости = 42; автоматических повторов нет'
if [[ -z "${MWS_API_KEY:-}" ]]; then
    read -r -s -p 'api-ключ сервисного аккаунта mws: ' MWS_API_KEY
    printf '\n'
    [[ -n "$MWS_API_KEY" ]] || { echo 'ключ не указан' >&2; exit 1; }
    export MWS_API_KEY
fi

"$python_bin" -m habr_search.evaluation.compare \
    --queries evaluation/queries_auto.csv --qrels evaluation/qrels_auto.csv \
    --qdrant-url http://127.0.0.1:6333 --output "$report_dir/comparison"
"$python_bin" -m habr_search.evaluation.evaluate_auto \
    --from-run "$report_dir/comparison" --qrels evaluation/qrels_auto.csv \
    --output "$report_dir/quality"
"$python_bin" -m habr_search.evaluation.benchmark_web \
    --queries evaluation/queries_auto_latency.csv --limit 0 --repeats 1 \
    --qdrant-url http://127.0.0.1:6333 --output "$report_dir/latency"
echo "качество и решения: $report_dir/quality"
echo "задержка http: $report_dir/latency"
