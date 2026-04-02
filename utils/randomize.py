import random
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from utils.exceptions import FrequencyConstraintError
from utils.frequency import coerce_float, get_row_controlled_pos, normalize_controlled_pos


@dataclass
class SamplingPolicy:
    seed: int = 0
    controlled_pos: Iterable[str] = field(default_factory=lambda: ("noun", "verb", "adjective"))
    zipf_min: Optional[Dict[str, Optional[float]]] = None
    zipf_max: Optional[Dict[str, Optional[float]]] = None
    zipf_weighted_sampling: bool = False
    zipf_temp: float = 1.0
    frequency_source: str = "wordfreq"
    overlay_enabled: bool = False
    overlay_path: Optional[str] = None
    overlay_manifest_path: Optional[str] = None
    frequency_cache_path: Optional[str] = None

    def __post_init__(self):
        self.controlled_pos = tuple(sorted(normalize_controlled_pos(self.controlled_pos)))
        self.zipf_min = {key: coerce_float(value) for key, value in dict(self.zipf_min or {}).items()}
        self.zipf_max = {key: coerce_float(value) for key, value in dict(self.zipf_max or {}).items()}

    def bounds_for(self, controlled_pos):
        return self.zipf_min.get(controlled_pos), self.zipf_max.get(controlled_pos)

    def summary(self):
        return {
            "frequency_source": self.frequency_source,
            "frequency_controlled_pos": list(self.controlled_pos),
            "frequency_zipf_min": dict(self.zipf_min),
            "frequency_zipf_max": dict(self.zipf_max),
            "frequency_weighted_sampling": self.zipf_weighted_sampling,
            "frequency_zipf_temp": self.zipf_temp,
            "frequency_overlay_enabled": self.overlay_enabled,
            "frequency_seed": self.seed,
        }


_ACTIVE_POLICY = None
_TRACE_EVENTS = []
_RNG = random.Random()


def configure_sampling_policy(policy: Optional[SamplingPolicy]):
    global _ACTIVE_POLICY, _TRACE_EVENTS, _RNG
    _ACTIVE_POLICY = policy
    _TRACE_EVENTS = []
    if policy is None:
        _RNG = random.Random()
        return
    _RNG = random.Random(policy.seed)
    random.seed(policy.seed)
    np.random.seed(policy.seed)


def clear_sampling_policy():
    configure_sampling_policy(None)


def get_active_policy():
    return _ACTIVE_POLICY


def get_active_policy_summary():
    if _ACTIVE_POLICY is None:
        return {}
    return _ACTIVE_POLICY.summary()


def begin_record_trace():
    global _TRACE_EVENTS
    _TRACE_EVENTS = []


def consume_record_trace():
    global _TRACE_EVENTS
    events = list(_TRACE_EVENTS)
    _TRACE_EVENTS = []
    return events


def uniform_choice(set, avoid=np.array([])):
    candidates = np.setdiff1d(set, avoid)
    if len(candidates) == 0:
        raise IndexError("Cannot choose from an empty sequence")
    return _RNG.choice(list(candidates))


def _is_structured_rows(candidates):
    return hasattr(getattr(candidates, "dtype", None), "names") and candidates.dtype.names is not None


def _policy_candidates(candidates):
    policy = _ACTIVE_POLICY
    if policy is None or len(candidates) == 0 or not _is_structured_rows(candidates):
        return None, None, None
    positions = {get_row_controlled_pos(row) for row in candidates}
    positions.discard(None)
    if len(positions) != 1:
        return None, None, None
    controlled_pos = next(iter(positions))
    if controlled_pos not in set(policy.controlled_pos):
        return None, None, None
    from utils.vocab_table import get_row_frequency
    eligible = []
    for row in candidates:
        frequency = get_row_frequency(row)
        zipf_value = coerce_float(frequency.get("zipf_expression")) or 0.0
        lower, upper = policy.bounds_for(controlled_pos)
        if lower is not None and zipf_value < lower:
            continue
        if upper is not None and zipf_value > upper:
            continue
        eligible.append((row, frequency))
    if not eligible:
        lower, upper = policy.bounds_for(controlled_pos)
        raise FrequencyConstraintError(
            "No %s candidates satisfy zipf bounds min=%s max=%s from %d candidates"
            % (controlled_pos, lower, upper, len(candidates))
        )
    return controlled_pos, eligible, policy


def _weighted_choice(eligible, policy):
    if not policy.zipf_weighted_sampling or len(eligible) == 1:
        return _RNG.choice(eligible)
    weights = []
    for _row, frequency in eligible:
        zipf_value = coerce_float(frequency.get("zipf_expression")) or 0.0
        base = 10 ** zipf_value if zipf_value > 0.0 else 0.001
        if policy.zipf_temp and policy.zipf_temp > 0:
            base = base ** (1.0 / policy.zipf_temp)
        weights.append(base if base > 0 else 0.001)
    return _RNG.choices(eligible, weights=weights, k=1)[0]


def _record_trace(row, frequency, controlled_pos, candidate_count, eligible_count, policy):
    if policy is None:
        return
    from utils.vocab_table import get_family_expressions, get_row_metadata
    metadata = get_row_metadata(row)
    event = {
        "expression": str(row["expression"]),
        "controlled_pos": controlled_pos,
        "zipf": coerce_float(frequency.get("zipf_expression")) or 0.0,
        "source": metadata.get("source", "base"),
        "_match_expressions": list(get_family_expressions(row) or [str(row["expression"])]),
    }
    if event["source"] != "base":
        event["source_lexicon"] = metadata.get("source_lexicon", "unknown")
        event["source_lemma"] = metadata.get("source_lemma", str(row["expression"]))
        if metadata.get("inherited_template"):
            event["inherited_template"] = metadata.get("inherited_template")
    _TRACE_EVENTS.append(event)

def decision(probability):
    return _RNG.random() < probability

def subset(set, probability):
    np.random.shuffle(set)
    return set[:math.floor(len(set) * probability)]

def choice(set, avoid=np.array([])):
    """
    :param set: a set of vocab entries
    :param avoid: list of vocab entries to avoid sampling
    :return: a random element from the set
    """
    candidates = np.setdiff1d(set, avoid)
    if len(candidates) == 0:
        raise IndexError("Cannot choose from an empty sequence")
    controlled_pos, eligible, policy = _policy_candidates(candidates)
    if eligible is None:
        return uniform_choice(candidates)
    row, frequency = _weighted_choice(eligible, policy)
    _record_trace(row, frequency, controlled_pos, len(candidates), len(eligible), policy)
    return row
