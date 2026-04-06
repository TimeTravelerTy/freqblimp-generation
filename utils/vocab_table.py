import json
import os
import re
from collections import defaultdict
from functools import lru_cache

import numpy as np

from utils.data_type import data_type
from utils.frequency import row_signature, zipf_for_expression


PROJECT_ROOT = "/".join(os.path.join(os.path.dirname(os.path.abspath(__file__))).split("/")[:-1])
BASE_VOCAB_PATH = os.path.join(PROJECT_ROOT, "vocabulary.csv")
DEFAULT_OVERLAY_PATH = os.path.join(PROJECT_ROOT, "vocabulary_overlay.csv")
DEFAULT_OVERLAY_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "vocabulary_overlay_manifest.json")
DEFAULT_FREQUENCY_CACHE_PATH = os.path.join(PROJECT_ROOT, "outputs", "cache", "vocabulary_frequency_cache.json")
_GET_ALL_CACHE = {}
_GET_ALL_CONJ_CACHE = {}
_GET_MATCHES_OF_CACHE = {}
_GET_MATCHED_BY_CACHE = {}


def clear_query_caches():
    _GET_ALL_CACHE.clear()
    _GET_ALL_CONJ_CACHE.clear()
    _GET_MATCHES_OF_CACHE.clear()
    _GET_MATCHED_BY_CACHE.clear()


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _decode_entry(entry):
    entry = entry.copy()
    entry[0] = re.sub("!", "'", entry[0])
    return entry


def _load_vocab_csv(path):
    if not path or not os.path.exists(path):
        return np.array([], dtype=data_type)
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=data_type)
    if getattr(table, "shape", ()) == ():
        table = np.array([table], dtype=data_type)
    return np.array([_decode_entry(entry) for entry in table], dtype=data_type)


def _load_overlay_manifest(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        payload = json.load(open(path))
    except Exception:
        return {}
    if isinstance(payload, dict):
        entries = payload.get("rows", [])
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []
    registry = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        signature = entry.get("row_signature")
        if signature:
            registry[str(signature)] = dict(entry)
    return registry


def _build_base_row_metadata(table):
    return {
        row_signature(row): {
            "row_signature": row_signature(row),
            "source": "base",
            "source_lexicon": "blimp",
            "source_lemma": str(row["expression"]),
            "inherited_template": None,
            "validation_status": "base",
        }
        for row in table
    }


def _merge_metadata(base_metadata, overlay_metadata):
    merged = dict(base_metadata)
    merged.update(overlay_metadata)
    return merged


def _build_family_expressions(table):
    by_root = defaultdict(set)
    by_expression = defaultdict(set)
    for row in table:
        expression = str(row["expression"]).strip()
        if expression:
            by_expression[expression].add(expression)
        root = str(row["root"]).strip()
        if root:
            by_root[root].add(expression)
        singularform = str(row["singularform"]).strip()
        pluralform = str(row["pluralform"]).strip()
        if singularform:
            by_expression[expression].add(singularform)
        if pluralform:
            by_expression[expression].add(pluralform)
    family = {}
    for row in table:
        signature = row_signature(row)
        expressions = set()
        expression = str(row["expression"]).strip()
        if expression:
            expressions.update(by_expression.get(expression, {expression}))
        root = str(row["root"]).strip()
        if root:
            expressions.update(by_root.get(root, set()))
        singularform = str(row["singularform"]).strip()
        pluralform = str(row["pluralform"]).strip()
        if singularform:
            expressions.add(singularform)
        if pluralform:
            expressions.add(pluralform)
        family[signature] = tuple(sorted(expr for expr in expressions if expr))
    return family


def _build_family_rows(table):
    by_root = defaultdict(list)
    by_expression = defaultdict(list)
    for row in table:
        expression = str(row["expression"]).strip()
        if expression:
            by_expression[expression].append(row)
        root = str(row["root"]).strip()
        if root:
            by_root[root].append(row)
    family_rows = {}
    for row in table:
        signature = row_signature(row)
        members = []
        seen = set()

        def _extend(rows_to_add):
            for candidate in rows_to_add:
                candidate_signature = row_signature(candidate)
                if candidate_signature in seen:
                    continue
                seen.add(candidate_signature)
                members.append(candidate)

        expression = str(row["expression"]).strip()
        if expression:
            _extend(by_expression.get(expression, ()))
        root = str(row["root"]).strip()
        if root:
            _extend(by_root.get(root, ()))
        singularform = str(row["singularform"]).strip()
        pluralform = str(row["pluralform"]).strip()
        if singularform:
            _extend(by_expression.get(singularform, ()))
        if pluralform:
            _extend(by_expression.get(pluralform, ()))
        family_rows[signature] = tuple(members)
    return family_rows


def _lemma_expression_for_row(row, family_rows):
    if row["verb"] == "1":
        bare_forms = [candidate for candidate in family_rows if candidate["bare"] == "1"]
        if bare_forms:
            bare_forms.sort(key=lambda candidate: str(candidate["expression"]))
            return str(bare_forms[0]["expression"]).strip()
        non_3sg_present = [candidate for candidate in family_rows if candidate["pres"] == "1" and candidate["3sg"] == "0"]
        if non_3sg_present:
            non_3sg_present.sort(key=lambda candidate: str(candidate["expression"]))
            return str(non_3sg_present[0]["expression"]).strip()
    if row["noun"] == "1":
        singular_forms = [candidate for candidate in family_rows if candidate["sg"] == "1" and candidate["pl"] != "1"]
        if singular_forms:
            singular_forms.sort(key=lambda candidate: str(candidate["expression"]))
            return str(singular_forms[0]["expression"]).strip()
    expression = str(row["expression"]).strip()
    if expression:
        return expression
    if family_rows:
        return str(family_rows[0]["expression"]).strip()
    return ""


def build_frequency_cache(table):
    family_expressions = _build_family_expressions(table)
    family_rows = _build_family_rows(table)
    cache = {}
    for row in table:
        signature = row_signature(row)
        expression = str(row["expression"]).strip()
        family = family_expressions.get(signature, ())
        lemma_expression = _lemma_expression_for_row(row, family_rows.get(signature, ()))
        family_zipf = [zipf_for_expression(expr) for expr in family if expr]
        family_zipf = [value for value in family_zipf if value > 0.0]
        cache[signature] = {
            "row_signature": signature,
            "expression": expression,
            "zipf_expression": zipf_for_expression(expression),
            "lemma_expression": lemma_expression,
            "zipf_lemma": zipf_for_expression(lemma_expression),
            "zipf_root": min(family_zipf) if family_zipf else zipf_for_expression(expression),
        }
    return cache


def write_frequency_cache(cache, path=DEFAULT_FREQUENCY_CACHE_PATH):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    payload = {
        "rows": list(cache.values()),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _load_frequency_cache(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        payload = json.load(open(path))
    except Exception:
        return {}
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    cache = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        signature = row.get("row_signature")
        if signature:
            cache[str(signature)] = dict(row)
    return cache


def _build_runtime_vocab():
    base_vocab = _load_vocab_csv(BASE_VOCAB_PATH)
    if _env_flag("FREQBLIMP_USE_OVERLAY", default=False):
        overlay_path = os.environ.get("FREQBLIMP_VOCAB_OVERLAY", DEFAULT_OVERLAY_PATH)
        overlay_vocab = _load_vocab_csv(overlay_path)
        if len(overlay_vocab) > 0:
            return np.concatenate([base_vocab, overlay_vocab]).astype(data_type)
    return base_vocab


def _filter_runtime_vocab(table):
    return np.array(list(filter(lambda x: x["OOV_inductive_biases"] != "1", table)), dtype=table.dtype)


_RAW_RUNTIME_VOCAB = _build_runtime_vocab()
vocab = _filter_runtime_vocab(_RAW_RUNTIME_VOCAB)
_OVERLAY_MANIFEST = _load_overlay_manifest(os.environ.get("FREQBLIMP_OVERLAY_MANIFEST", DEFAULT_OVERLAY_MANIFEST_PATH))
ROW_METADATA_BY_SIGNATURE = _merge_metadata(_build_base_row_metadata(vocab), _OVERLAY_MANIFEST)
ROW_FAMILY_EXPRESSIONS_BY_SIGNATURE = _build_family_expressions(vocab)
FREQUENCY_CACHE = _load_frequency_cache(os.environ.get("FREQBLIMP_FREQUENCY_CACHE", DEFAULT_FREQUENCY_CACHE_PATH))
if len(FREQUENCY_CACHE) < len(vocab):
    FREQUENCY_CACHE = build_frequency_cache(vocab)


def get_runtime_vocab():
    return vocab


def get_row_metadata(row):
    return ROW_METADATA_BY_SIGNATURE.get(row_signature(row), {})


def get_row_frequency(row):
    signature = row_signature(row)
    record = FREQUENCY_CACHE.get(signature)
    if record is None:
        record = build_frequency_cache(np.array([row], dtype=vocab.dtype)).get(signature, {})
    return record


def get_family_expressions(row):
    return tuple(ROW_FAMILY_EXPRESSIONS_BY_SIGNATURE.get(row_signature(row), ()))


def get_frequency_cache():
    return dict(FREQUENCY_CACHE)


def _table_cache_key(table):
    interface = getattr(table, "__array_interface__", None)
    data_ptr = None
    if interface and interface.get("data"):
        data_ptr = interface["data"][0]
    return data_ptr, getattr(table, "shape", None), getattr(table, "strides", None), str(getattr(table, "dtype", ""))


def get_all(label, value, table=vocab):
    """
    :param label: string. field name.
    :param value: string. label.
    :param table: ndarray of vocab items.
    :return: table restricted to all entries with "value" in field "label"
    """
    key = (label, value, _table_cache_key(table))
    if key not in _GET_ALL_CACHE:
        # TODO: this should not be based on string equality, but disjunction matching
        # return np.array(list(filter(lambda x: condition_is_match_disj(value, x[label]), table)), dtype=data_type)
        _GET_ALL_CACHE[key] = np.array(list(filter(lambda x: x[label] == value, table)), dtype=table.dtype)
    return _GET_ALL_CACHE[key]

def get_all_conjunctive(labels_values, table=vocab):
    """
    :param labels_values: list of (l,v) pairs: [(l1, v1), (l2, v2), (l3, v3)]
    :return: vocab items with the given value for each label
    """
    key = (tuple(labels_values), _table_cache_key(table))
    if key not in _GET_ALL_CONJ_CACHE:
        to_return = table
        for label, value in labels_values:
            to_return = np.array(list(filter(lambda x: x[label] == value, to_return)), dtype=table.dtype)
        _GET_ALL_CONJ_CACHE[key] = to_return
    return _GET_ALL_CONJ_CACHE[key]


def get_matches_of(row, label, table=vocab):
    """
    :param row: ndarray row. functor vocab item.
    :param label: string. field containing selectional restrictions.
    :param table: ndarray of vocab items.
    :return: all entries in table that match the selectional restrictions of row as given in label.
    """
    value = str(np.array(row, dtype=table.dtype)[label])
    if value == "":
        pass
    else:
        key = (row_signature(np.array(row, dtype=table.dtype)), label, _table_cache_key(table))
        if key not in _GET_MATCHES_OF_CACHE:
            matches = []
            values = str(value).split(";")
            for disjunct in values:
                k_vs = conj_list(disjunct)
                matches.extend(list(get_all_conjunctive(k_vs, table)))
            _GET_MATCHES_OF_CACHE[key] = np.array(matches, dtype=table.dtype)
        return _GET_MATCHES_OF_CACHE[key]


def get_matches_of_conj(rows_labels, table=vocab):
    """
    :param rows_labels: list of (r,l) pairs: [(r1, l1), (r2, l2), (r3, l3)]
    :param table: ndarray of vocab items.
    :return: all entries in table that match the selectional restrictions of all rows as given by labels.
    """
    to_return = table
    for row, label in rows_labels:
        value = str(np.array(row, dtype=table.dtype)[label])
        if value == "":
            pass
        else:
            to_return = np.array(list(filter(lambda x: is_match_disj(x, value), to_return)), dtype=table.dtype)
    return to_return


def get_matched_by(row, label, table=vocab):
    """
    :param row: ndarray row. selected vocab item.
    :param label: string. field containing selectional restrictions.
    :param table: ndarray of vocab items.
    :return: all entries in table whose selectional restrictions in label are matched by row.
    """
    key = (row_signature(np.array(row, dtype=table.dtype)), label, _table_cache_key(table))
    if key not in _GET_MATCHED_BY_CACHE:
        matches = []
        for entry in table:
            value = str(np.array(entry, dtype=table.dtype)[label])
            if is_match_disj(row, value):
                matches.append(entry)
        _GET_MATCHED_BY_CACHE[key] = np.array(matches, dtype=table.dtype)
    return _GET_MATCHED_BY_CACHE[key]


@lru_cache(maxsize=None)
def conj_list(conjunction):
    """
    :param disjunct: a string corresponding to a conjunction of selectional restrictions
    :return: a list of k, v pairs 
    """
    try:
        to_return = tuple((v.split("=")[0], v.split("=")[1]) for v in conjunction.split("^"))
        return to_return
    except IndexError:
        pass

def is_match_disj(row, disjunction):
    """
    :param row: a vocab item
    :param disjunction: a string corresponding to a disjunction of selectional restrictions
    :return: true if the row matches one of the disjuncts, false otherwise
    """
    if disjunction == "":
        return True
    else:
        disjuncts = disjunction.split(";")
        match = False
        for d in disjuncts:
            match = match or is_match_conj(row, d)
        return match

def is_match_conj(row, conjunction):
    """
    :param row: a vocab item
    :param conjunction: a string corresponding to a conjunction of selectional restrictions
    :return: true if the row matches the conjunction, false otherwise
    """
    conjuncts = conj_list(conjunction)
    match = True
    for k, v in conjuncts:
        try:
            match = match and row[k] == v
        except TypeError:
            pass
    return match

def condition_is_match_disj(condition, disjunction):
    """
    :param condition: a string representing a selectional condition
    :param disjunction: a string corresponding to a disjunction of selectional restrictions
    :return: true if the row matches one of the disjuncts, false otherwise
    """
    if disjunction == "":
        return True
    else:
        disjuncts = disjunction.split(";")
        match = False
        for d in disjuncts:
            match = match or condition_is_match_conj(condition, d)
        return match

def condition_is_match_conj(condition, conjunction):
    """
    :param condition: a string representing a selectional condition
    :param conjunction: a string corresponding to a conjunction of selectional restrictions
    :return: true if the row matches the conjunction, false otherwise
    """
    conjuncts = conj_list(conjunction)
    match = True
    for k, v in conjuncts:
        try:
            match = match and condition[k] == v
        except TypeError:
            pass
    return match
