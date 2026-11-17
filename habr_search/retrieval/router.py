"""выбираем модель или гибрид по проверяемым признакам запроса и заголовков."""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

ROUTER_VERSION: str = 'rules-v1'
WORDS: re.Pattern[str] = re.compile(r'[^\W_]+', re.UNICODE)
IDENTIFIERS: re.Pattern[str] = re.compile(r'(?<!\w)[A-Za-z][A-Za-z0-9_]*(?:[.:/+\-][A-Za-z0-9_]+)*(?!\w)')
QUOTED: re.Pattern[str] = re.compile(r'["«]([^"»]{3,100})["»]')
STOP_WORDS: frozenset[str] = frozenset(
    'как что это для при или или же из на в с со и по к у о об от до без через средствами '
    'можно нужно помощью использовать работает работа пример примеры сделать задачи какие '
    'the a an of to in with how what use using and or'.split())


@dataclass(frozen=True)
class RouterConfig:
    """фиксируем начальные правила; их пороги ещё требуют проверки на новых темах."""
    min_title_coverage: float = 0.6
    min_lexical_gap: float = 0.1
    lexical_wait_ms: float = 5.0  # дополнительное ожидание после готовности модели

    def __post_init__(self) -> None:
        for value in (self.min_title_coverage, self.min_lexical_gap):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('пороги маршрутизатора должны быть от 0 до 1')
        if not math.isfinite(self.lexical_wait_ms) or self.lexical_wait_ms < 0:
            raise ValueError('auto-lexical-wait-ms должен быть конечным неотрицательным числом')


@dataclass(frozen=True)
class RoutingHit:
    article_id: int
    title: str
    score: float


@dataclass(frozen=True)
class RoutingFeatures:
    identifiers: tuple[str, ...] = ()
    exact_title_match: bool = False
    title_coverage: float = 0.0
    lexical_gap: float = 0.0
    top5_overlap: int = 0


@dataclass(frozen=True)
class RoutingDecision:
    engine: Literal['bge-m3', 'hybrid', 'ltr']
    reason: str
    features: RoutingFeatures = RoutingFeatures()
    version: str = ROUTER_VERSION


def normalize(text: str) -> str:
    return text.casefold().replace('ё', 'е')


def query_identifiers(query: str) -> tuple[str, ...]:
    """выделяем технические обозначения и цитаты, не привязываясь к статьям корпуса."""
    found = [match[0] for match in IDENTIFIERS.finditer(query)
             if len(match[0]) >= 3 and (any(char.isdigit() for char in match[0])
                or any(char in match[0] for char in '_.:/-') or re.search(r'[a-z][A-Z]', match[0]))]
    found.extend(match[1].strip() for match in QUOTED.finditer(query) if match[1].strip())
    return tuple(dict.fromkeys(normalize(item) for item in found))


def choose_route(query: str, dense: Sequence[int], lexical: Sequence[RoutingHit],
                 config: RouterConfig = RouterConfig()) -> RoutingDecision:
    """усиливаем лексическую ветку только при нескольких согласующихся признаках."""
    identifiers = query_identifiers(query)
    terms = set(WORDS.findall(normalize(query))) - STOP_WORDS
    first = lexical[0] if lexical else None
    title = normalize(first.title) if first is not None else ''
    title_terms = set(WORDS.findall(title))
    # границы защищают от ложных совпадений вроде python3 в python310
    exact = any(re.search(r'(?<!\w)' + re.escape(item) + r'(?!\w)', title) for item in identifiers)
    coverage = len(terms & title_terms) / len(terms) if terms else 0.0
    score = first.score if first is not None else 0.0
    second = lexical[1].score if len(lexical) > 1 else 0.0
    gap = max(0.0, (score - second) / score) if score > 0 else 0.0
    overlap = len(set(dense[:5]) & {item.article_id for item in lexical[:5]})
    features = RoutingFeatures(identifiers, exact, coverage, gap, overlap)
    if not lexical:
        reason = 'no_lexical_matches'
    elif not identifiers:
        reason = 'semantic_query'
    elif not exact:
        reason = 'identifier_not_in_top_title'
    elif coverage < config.min_title_coverage:
        reason = 'insufficient_title_coverage'
    elif gap < config.min_lexical_gap:
        reason = 'ambiguous_lexical_results'
    elif dense and first is not None and dense[0] == first.article_id:
        reason = 'exact_answer_already_first'
    else:
        return RoutingDecision('hybrid', 'specific_title_match', features)
    return RoutingDecision('bge-m3', reason, features)
