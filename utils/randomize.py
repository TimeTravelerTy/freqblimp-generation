import random
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from utils.exceptions import FrequencyConstraintError, LexicalGapError
from utils.frequency import coerce_float, normalize_controlled_pos


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
        self.controlled_pos_set = set(self.controlled_pos)
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
_TRACE_ENABLED = True
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


def set_trace_recording_enabled(enabled: bool):
    global _TRACE_ENABLED, _TRACE_EVENTS
    _TRACE_ENABLED = bool(enabled)
    if not _TRACE_ENABLED:
        _TRACE_EVENTS = []


def trace_recording_enabled():
    return _TRACE_ENABLED


def begin_record_trace():
    global _TRACE_EVENTS
    if not _TRACE_ENABLED:
        return
    _TRACE_EVENTS = []


def consume_record_trace():
    global _TRACE_EVENTS
    if not _TRACE_ENABLED:
        return []
    events = list(_TRACE_EVENTS)
    _TRACE_EVENTS = []
    return events


def _without_avoid(candidates, avoid):
    if avoid is None:
        return candidates
    try:
        if len(avoid) == 0:
            return candidates
    except TypeError:
        pass
    return np.setdiff1d(candidates, avoid)


def uniform_choice(set, avoid=None):
    candidates = _without_avoid(set, avoid)
    if len(candidates) == 0:
        raise LexicalGapError("No candidates available for uniform sampling")
    try:
        return candidates[_RNG.randrange(len(candidates))]
    except (TypeError, KeyError, IndexError):
        pass
    return _RNG.choice(tuple(candidates))


def _is_structured_rows(candidates):
    return hasattr(getattr(candidates, "dtype", None), "names") and candidates.dtype.names is not None


def _candidate_controlled_pos(candidates):
    if not _is_structured_rows(candidates) or len(candidates) == 0:
        return None
    names = candidates.dtype.names
    if "noun" in names and np.all(candidates["noun"] == "1"):
        return "noun"
    if "verb" in names and np.all(candidates["verb"] == "1"):
        return "verb"
    if "category_2" in names and np.all(np.isin(candidates["category_2"], ("Adj", "adjective"))):
        return "adjective"
    return None


def _policy_candidates(candidates):
    policy = _ACTIVE_POLICY
    if policy is None or len(candidates) == 0 or not _is_structured_rows(candidates):
        return None, None, None, None, None
    controlled_pos = _candidate_controlled_pos(candidates)
    if controlled_pos is None:
        return None, None, None, None, None
    if controlled_pos not in policy.controlled_pos_set:
        return None, None, None, None, None
    from utils.vocab_table import get_table_zipf_expression
    zipf_values = get_table_zipf_expression(candidates)
    lower, upper = policy.bounds_for(controlled_pos)
    eligible_mask = np.ones(len(candidates), dtype=bool)
    if lower is not None:
        eligible_mask &= zipf_values >= lower
    if upper is not None:
        eligible_mask &= zipf_values <= upper
    if not np.any(eligible_mask):
        lower, upper = policy.bounds_for(controlled_pos)
        raise FrequencyConstraintError(
            "No %s candidates satisfy zipf bounds min=%s max=%s from %d candidates"
            % (controlled_pos, lower, upper, len(candidates))
        )
    eligible_indices = np.flatnonzero(eligible_mask)
    return controlled_pos, candidates, eligible_indices, zipf_values[eligible_indices], policy


def _weighted_choice(candidates, eligible_indices, eligible_zipf, policy):
    if len(eligible_indices) == 1:
        return candidates[eligible_indices[0]], float(eligible_zipf[0])
    if not policy.zipf_weighted_sampling:
        position = _RNG.randrange(len(eligible_indices))
        return candidates[eligible_indices[position]], float(eligible_zipf[position])
    weights = []
    for zipf_value in eligible_zipf:
        base = 10 ** float(zipf_value) if float(zipf_value) > 0.0 else 0.001
        if policy.zipf_temp and policy.zipf_temp > 0:
            base = base ** (1.0 / policy.zipf_temp)
        weights.append(base if base > 0 else 0.001)
    position = _RNG.choices(range(len(eligible_indices)), weights=weights, k=1)[0]
    return candidates[eligible_indices[position]], float(eligible_zipf[position])


def _record_trace(row, frequency, controlled_pos, candidate_count, eligible_count, policy):
    if policy is None or not _TRACE_ENABLED:
        return
    from utils.vocab_table import get_family_expressions, get_row_metadata
    metadata = get_row_metadata(row)

    event = {
        "expression": str(row["expression"]),
        "pos": controlled_pos,
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

def choice(set, avoid=None):
    """
    :param set: a set of vocab entries
    :param avoid: list of vocab entries to avoid sampling
    :return: a random element from the set
    """
    candidates = _without_avoid(set, avoid)
    if len(candidates) == 0:
        raise LexicalGapError("No candidates available for policy-aware sampling")
    controlled_pos, policy_candidates, eligible_indices, eligible_zipf, policy = _policy_candidates(candidates)
    if policy_candidates is None:
        return uniform_choice(candidates)
    row, zipf_value = _weighted_choice(policy_candidates, eligible_indices, eligible_zipf, policy)
    frequency = {"zipf_expression": zipf_value}
    _record_trace(row, frequency, controlled_pos, len(candidates), len(eligible_indices), policy)
    return row
