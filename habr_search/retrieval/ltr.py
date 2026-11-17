"""загружаем текстовую модель lightgbm и проверяем её совместимость с индексами."""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from habr_search.retrieval.hybrid import FusedRank
from habr_search.retrieval.ltr_features import (
    FEATURE_NAMES, FEATURE_VERSION, CandidateFeatures, FeatureCatalog, RankSignal, build_features,
)

if TYPE_CHECKING:
    from lightgbm import Booster
    import numpy as np
    from numpy.typing import NDArray


class BoosterAPI(Protocol):
    """уточняем неполные numpy-аннотации lightgbm на границе предсказания."""
    def predict(self, data: NDArray[np.float64], *, num_threads: int) -> NDArray[np.float64]: ...
    def feature_importance(self, importance_type: str) -> Sequence[float]: ...


def require_lightgbm() -> None:
    """обычные режимы не импортируют необязательные numpy и lightgbm."""
    try:
        importlib.import_module('lightgbm')
    except (ImportError, OSError) as exc:
        raise RuntimeError('для ltr установите python -m pip install -r requirements-ltr.txt') from exc


def predict_candidates(booster: Booster, rows: Sequence[CandidateFeatures], limit: int) -> list[FusedRank]:
    """общий путь предсказания для фолдов проверки и загруженного серверного артефакта."""
    if not rows:
        return []
    import numpy as np
    matrix: NDArray[np.float64] = np.asarray([row.values for row in rows], dtype=np.float64)
    predicted = cast(BoosterAPI, booster).predict(matrix, num_threads=1)
    if predicted.ndim != 1 or len(predicted) != len(rows):
        raise ValueError('lightgbm вернул неверное число оценок')
    scores = [float(value) for value in predicted]
    if any(not math.isfinite(score) for score in scores):
        raise ValueError('lightgbm вернул неконечную оценку')
    found = [FusedRank(row.article_id, score, row.dense_rank, row.lexical_rank) for row, score in zip(rows, scores)]
    absent = limit + 1
    found.sort(key=lambda item: (-item.score, item.dense_rank or absent, item.lexical_rank or absent, item.article_id))
    return found


class LearnedRanker:
    """одна загруженная модель обслуживает все запросы без повторного обучения."""
    def __init__(self, directory: Path) -> None:
        require_lightgbm()
        import lightgbm as lgb
        decoded: object = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        if not isinstance(decoded, dict):
            raise ValueError('повреждены метаданные ltr')
        data = cast(dict[str, object], decoded)
        if (data.get('feature_version') != FEATURE_VERSION or data.get('feature_names') != list(FEATURE_NAMES)
                or data.get('objective') != 'lambdarank' or data.get('format_version') != 1):
            raise ValueError('несовместимая схема признаков или формат модели ltr')
        limit = data.get('candidate_limit')
        if type(limit) is not int or not 20 <= limit <= 100:
            raise ValueError('повреждена глубина кандидатов ltr')
        self.candidate_limit: int = limit
        self.metadata: dict[str, object] = data
        self.model_hash: str = hashlib.sha256((directory / 'model.txt').read_bytes()).hexdigest()
        if data.get('model_sha256') != self.model_hash:
            raise ValueError('контрольная сумма модели ltr не совпала')
        self.booster: Booster = lgb.Booster(model_file=str(directory / 'model.txt'))
        if self.booster.feature_name() != list(FEATURE_NAMES):
            raise ValueError('порядок признаков модели ltr отличается от кода')

    def validate_context(self, collection: str, qdrant_revision: str, tantivy_revision: str,
                         candidate_limit: int) -> None:
        """модель переносится вместе с метаданными; подмена индекса требует переобучения."""
        if any(self.metadata.get(key) != value for key, value in (
            ('collection', collection), ('qdrant_revision', qdrant_revision),
            ('tantivy_revision', tantivy_revision), ('candidate_limit', candidate_limit),
        )):
            raise ValueError('индексы или глубина кандидатов отличаются от обучения ltr')

    def predict(self, rows: Sequence[CandidateFeatures]) -> list[FusedRank]:
        """один поток предсказания ограничивает накладные расходы openmp на коротких списках."""
        return predict_candidates(self.booster, rows, self.candidate_limit)

    def rank(self, query: str, dense: Sequence[RankSignal], lexical: Sequence[RankSignal],
             catalog: FeatureCatalog) -> list[FusedRank]:
        return self.predict(build_features(query, dense, lexical, catalog, self.candidate_limit))
