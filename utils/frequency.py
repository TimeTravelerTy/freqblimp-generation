import math
import re
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Set

from wordfreq import zipf_frequency


CONTROLLED_POS = ("noun", "verb", "adjective")
_ADJECTIVE_TAGS = {"Adj", "adjective"}
_PUNCT_RE = re.compile(r"^[^a-z0-9]+|[^a-z0-9]+$")


def normalize_controlled_pos(values: Optional[Iterable[str]]) -> Set[str]:
    if values is None:
        return set(CONTROLLED_POS)
    normalized = set()
    for value in values:
        if value is None:
            continue
        key = str(value).strip().lower()
        if key in {"noun", "nouns"}:
            normalized.add("noun")
        elif key in {"verb", "verbs"}:
            normalized.add("verb")
        elif key in {"adjective", "adjectives", "adj", "adjs"}:
            normalized.add("adjective")
    return normalized


def get_row_controlled_pos(row) -> Optional[str]:
    names = getattr(getattr(row, "dtype", None), "names", None)
    if not names:
        return None
    if "noun" in names and row["noun"] == "1":
        return "noun"
    if "verb" in names and row["verb"] == "1":
        return "verb"
    if "category_2" in names and row["category_2"] in _ADJECTIVE_TAGS:
        return "adjective"
    return None


def row_signature(row) -> str:
    names = getattr(getattr(row, "dtype", None), "names", None)
    if not names:
        return str(row)
    return "||".join("%s=%s" % (name, row[name]) for name in names)


def zipf_for_expression(expression: str) -> float:
    if expression is None:
        return 0.0
    text = str(expression).strip().lower()
    if not text:
        return 0.0
    try:
        return float(zipf_frequency(text, "en"))
    except Exception:
        return 0.0


def values_to_aggregates(values: Sequence[float]) -> Dict[str, Optional[float]]:
    positives = [float(value) for value in values if float(value) > 0.0]
    oov_count = len(values) - len(positives)
    if not positives:
        return {
            "n": 0,
            "oov_count": oov_count,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "n": len(positives),
        "oov_count": oov_count,
        "mean": sum(positives) / len(positives),
        "median": float(statistics.median(positives)),
        "min": float(min(positives)),
        "max": float(max(positives)),
    }


def normalize_sentence_tokens(sentence: str) -> List[str]:
    tokens = []
    for raw_token in str(sentence or "").lower().split():
        token = _PUNCT_RE.sub("", raw_token)
        if token:
            tokens.append(token)
    return tokens


def sentence_contains_expression(sentence: str, expression: str) -> bool:
    expr_tokens = normalize_sentence_tokens(expression)
    if not expr_tokens:
        return False
    sent_tokens = normalize_sentence_tokens(sentence)
    if not sent_tokens:
        return False
    width = len(expr_tokens)
    for idx in range(0, len(sent_tokens) - width + 1):
        if sent_tokens[idx:idx + width] == expr_tokens:
            return True
    return False


def trace_events_for_sentence(sentence: str,
                              events: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    matched = []
    for event in events:
        family_expressions = list(event.get("_match_expressions", ()))
        if not family_expressions and event.get("expression"):
            family_expressions = [str(event["expression"])]
        if any(sentence_contains_expression(sentence, expr) for expr in family_expressions):
            matched.append({key: value for key, value in dict(event).items() if not key.startswith("_")})
    return matched


def trace_zipf_aggregates(events: Sequence[Dict[str, object]]) -> Dict[str, Optional[float]]:
    values = [float(event.get("zipf", event.get("zipf_expression", 0.0)) or 0.0) for event in events]
    return values_to_aggregates(values)


def coerce_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed
