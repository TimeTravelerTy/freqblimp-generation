import gc
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
_TABLE_LABEL_INDEX = {}  # (label, table_key) -> {str_value -> np.array of row indices}
_TABLE_ZIPF_EXPRESSION_CACHE = {}
_TABLE_ROW_SIGNATURE_CACHE = {}
_EXPRESSION_ZIPF_REGISTRY = None
_EXPRESSION_ZIPF_REGISTRY_PATH = None
_ROW_FREQUENCY_CACHE = {}
_LAZY_REGISTRY = []  # LazyVocabSet instances that should be reset between paradigms

# Max entries for caches that store large numpy array slices.
# Prevents unbounded memory growth when many unique (row, label, table) combinations
# are seen within a single paradigm run (e.g. with the large vocabulary overlay).
_RESULT_CACHE_MAX = 256


def clear_query_caches():
    global _OVERLAY_METADATA_REGISTRY, _OVERLAY_METADATA_REGISTRY_PATH
    global _EXPRESSION_ZIPF_REGISTRY, _EXPRESSION_ZIPF_REGISTRY_PATH
    _GET_ALL_CACHE.clear()
    _GET_ALL_CONJ_CACHE.clear()
    _GET_MATCHES_OF_CACHE.clear()
    _GET_MATCHED_BY_CACHE.clear()
    _TABLE_LABEL_INDEX.clear()
    _TABLE_ZIPF_EXPRESSION_CACHE.clear()
    _TABLE_ROW_SIGNATURE_CACHE.clear()
    _ROW_FREQUENCY_CACHE.clear()
    _OVERLAY_METADATA_REGISTRY = None
    _OVERLAY_METADATA_REGISTRY_PATH = None
    _EXPRESSION_ZIPF_REGISTRY = None
    _EXPRESSION_ZIPF_REGISTRY_PATH = None
    # Release cached vocab subsets held by LazyVocabSet instances so the large
    # numpy arrays can be freed between paradigm runs.
    for _lazy_set in _LAZY_REGISTRY:
        _lazy_set._value = None
    gc.collect()


def _get_label_index(label, table):
    """Build and cache an inverted index for a field of any table: value -> row indices."""
    tkey = (label, _table_cache_key(table))
    if tkey not in _TABLE_LABEL_INDEX:
        col = table[label]
        unique_vals, inverse = np.unique(col, return_inverse=True)
        sort_order = np.argsort(inverse, kind="stable")
        sorted_codes = inverse[sort_order]
        splits = np.where(np.diff(sorted_codes))[0] + 1
        groups = np.split(sort_order, splits)
        _TABLE_LABEL_INDEX[tkey] = {str(unique_vals[i]): groups[i] for i in range(len(unique_vals))}
    return _TABLE_LABEL_INDEX[tkey]


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
    npy_cache = path + ".npy"
    if os.path.exists(npy_cache) and os.path.getmtime(npy_cache) >= os.path.getmtime(path):
        try:
            return np.load(npy_cache, allow_pickle=False)
        except Exception:
            pass
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=data_type)
    if getattr(table, "shape", ()) == ():
        table = np.array([table], dtype=data_type)
    table = table.copy()
    table["expression"] = np.char.replace(table["expression"], "!", "'")
    try:
        np.save(npy_cache, table)
    except Exception:
        pass
    return table


def _load_overlay_manifest(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            payload = json.load(handle)
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


def _get_table_row_signatures(table):
    key = _table_cache_key(table)
    if key not in _TABLE_ROW_SIGNATURE_CACHE:
        _TABLE_ROW_SIGNATURE_CACHE[key] = tuple(row_signature(row) for row in table)
    return _TABLE_ROW_SIGNATURE_CACHE[key]


def _build_base_row_metadata(table):
    signatures = _get_table_row_signatures(table)
    return {
        signature: {
            "row_signature": signature,
            "source": "base",
            "source_lexicon": "blimp",
            "source_lemma": str(row["expression"]),
            "inherited_template": None,
            "validation_status": "base",
        }
        for row, signature in zip(table, signatures)
    }


def _merge_metadata(base_metadata, overlay_metadata):
    merged = dict(base_metadata)
    merged.update(overlay_metadata)
    return merged


def _build_family_expressions(table):
    by_root = defaultdict(set)
    by_expression = defaultdict(set)
    signatures = _get_table_row_signatures(table)
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
    for row, signature in zip(table, signatures):
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
    signatures = _get_table_row_signatures(table)
    for idx, row in enumerate(table):
        expression = str(row["expression"]).strip()
        if expression:
            by_expression[expression].append(idx)
        root = str(row["root"]).strip()
        if root:
            by_root[root].append(idx)
    family_rows = {}
    for idx, row in enumerate(table):
        signature = signatures[idx]
        member_indices = []
        seen_signatures = set()

        def _extend(indices_to_add):
            for candidate_idx in indices_to_add:
                candidate_signature = signatures[candidate_idx]
                if candidate_signature in seen_signatures:
                    continue
                seen_signatures.add(candidate_signature)
                member_indices.append(candidate_idx)

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
        family_rows[signature] = tuple(table[candidate_idx] for candidate_idx in member_indices)
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
    expressions = sorted(
        {
            str(expression).strip()
            for expression in np.asarray(table["expression"], dtype=str)
            if str(expression).strip()
        }
    )
    return {expression: zipf_for_expression(expression) for expression in expressions}


def write_frequency_cache(cache, path=DEFAULT_FREQUENCY_CACHE_PATH):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    payload = {
        "format_version": 2,
        "expressions": dict(cache),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def _load_frequency_cache(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except Exception:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get("expressions"), dict):
        registry = {}
        for expression, value in payload["expressions"].items():
            expression = str(expression).strip()
            if expression:
                registry[expression] = float(value or 0.0)
        return registry
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
        expression = str(row.get("expression", "")).strip()
        if expression and expression not in cache:
            cache[expression] = float(row.get("zipf_expression") or 0.0)
    return cache


def _filter_overlay_acronym_nouns(table):
    """Remove noun rows that look like lowercased acronyms (no standard vowels, e.g. 'nlp', 'mvp', 'psf')."""
    is_noun = table["noun"] == "1"
    if not np.any(is_noun):
        return table
    expressions = np.asarray(table["expression"], dtype=str)
    has_vowel = np.zeros(len(table), dtype=bool)
    for v in "aeiouAEIOU":
        has_vowel |= np.char.find(expressions, v) >= 0
    acronym_mask = is_noun & ~has_vowel
    return table[~acronym_mask]


def _build_runtime_vocab():
    base_vocab = _load_vocab_csv(BASE_VOCAB_PATH)
    if _env_flag("FREQBLIMP_USE_OVERLAY", default=False):
        overlay_path = os.environ.get("FREQBLIMP_VOCAB_OVERLAY", DEFAULT_OVERLAY_PATH)
        overlay_vocab = _load_vocab_csv(overlay_path)
        if len(overlay_vocab) > 0:
            overlay_vocab = _filter_overlay_acronym_nouns(overlay_vocab)
            # copy=False avoids an extra ~3.9 GB allocation when dtype already matches
            return np.concatenate([base_vocab, overlay_vocab]).astype(data_type, copy=False)
    return base_vocab


def _filter_runtime_vocab(table):
    return table[table["OOV_inductive_biases"] != "1"]


# Build and filter in one expression so the unfiltered intermediate table is
# not kept alive as a persistent module-level global (~3.9 GB with overlay).
vocab = _filter_runtime_vocab(_build_runtime_vocab())

# Lazy-loaded structures — only built on first access (avoids O(N) startup cost with large overlays)
_OVERLAY_METADATA_REGISTRY = None
_OVERLAY_METADATA_REGISTRY_PATH = None


def _ensure_overlay_metadata_registry():
    global _OVERLAY_METADATA_REGISTRY, _OVERLAY_METADATA_REGISTRY_PATH
    path = os.environ.get("FREQBLIMP_OVERLAY_MANIFEST", DEFAULT_OVERLAY_MANIFEST_PATH)
    if _OVERLAY_METADATA_REGISTRY is None or _OVERLAY_METADATA_REGISTRY_PATH != path:
        _OVERLAY_METADATA_REGISTRY = _load_overlay_manifest(path)
        _OVERLAY_METADATA_REGISTRY_PATH = path
    return _OVERLAY_METADATA_REGISTRY


def get_runtime_vocab():
    return vocab


def get_row_metadata(row):
    signature = row_signature(row)
    metadata = _ensure_overlay_metadata_registry().get(signature)
    if metadata is not None:
        return metadata
    expression = str(row["expression"]).strip()
    return {
        "row_signature": signature,
        "source": "base",
        "source_lexicon": "blimp",
        "source_lemma": expression,
        "inherited_template": None,
        "validation_status": "base",
    }


def get_row_frequency(row):
    signature = row_signature(row)
    cached = _ROW_FREQUENCY_CACHE.get(signature)
    if cached is not None:
        return dict(cached)

    expression = str(row["expression"]).strip()
    zipf_expression = _zipf_for_cached_expression(expression)
    record = {
        "row_signature": signature,
        "expression": expression,
        "zipf_expression": zipf_expression,
        "lemma_expression": expression,
        "zipf_lemma": zipf_expression,
        "zipf_root": zipf_expression,
    }

    root = str(row["root"]).strip()
    family_rows = tuple(get_all("root", root)) if root else (row,)
    if family_rows:
        record["lemma_expression"] = _lemma_expression_for_row(row, family_rows)
        record["zipf_lemma"] = _zipf_for_cached_expression(record["lemma_expression"])
        family_zipf = [
            _zipf_for_cached_expression(candidate["expression"])
            for candidate in family_rows
            if str(candidate["expression"]).strip()
        ]
        family_zipf = [value for value in family_zipf if value > 0.0]
        if family_zipf:
            record["zipf_root"] = min(family_zipf)

    if len(_ROW_FREQUENCY_CACHE) >= _RESULT_CACHE_MAX:
        _ROW_FREQUENCY_CACHE.pop(next(iter(_ROW_FREQUENCY_CACHE)))
    _ROW_FREQUENCY_CACHE[signature] = dict(record)
    return record


def get_family_expressions(row):
    expression = str(row["expression"]).strip()
    return (expression,) if expression else ()


def get_frequency_cache():
    return dict(_ensure_expression_zipf_registry())


def _ensure_expression_zipf_registry():
    global _EXPRESSION_ZIPF_REGISTRY, _EXPRESSION_ZIPF_REGISTRY_PATH
    path = os.environ.get("FREQBLIMP_FREQUENCY_CACHE", DEFAULT_FREQUENCY_CACHE_PATH)
    if _EXPRESSION_ZIPF_REGISTRY is None or _EXPRESSION_ZIPF_REGISTRY_PATH != path:
        _EXPRESSION_ZIPF_REGISTRY = _load_frequency_cache(path)
        _EXPRESSION_ZIPF_REGISTRY_PATH = path
    return _EXPRESSION_ZIPF_REGISTRY


def _zipf_for_cached_expression(expression):
    expression = str(expression).strip()
    if not expression:
        return 0.0
    registry = _ensure_expression_zipf_registry()
    if expression not in registry:
        registry[expression] = zipf_for_expression(expression)
    return float(registry[expression])


def get_table_zipf_expression(table):
    key = _table_cache_key(table)
    if key not in _TABLE_ZIPF_EXPRESSION_CACHE:
        expressions = np.asarray(table["expression"], dtype=str)
        unique_expressions, inverse = np.unique(expressions, return_inverse=True)
        unique_zipf = np.array(
            [_zipf_for_cached_expression(expression) for expression in unique_expressions],
            dtype=np.float32,
        )
        _TABLE_ZIPF_EXPRESSION_CACHE[key] = unique_zipf[inverse]
    return _TABLE_ZIPF_EXPRESSION_CACHE[key]


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
        idx = _get_label_index(label, table)
        indices = idx.get(str(value))
        _GET_ALL_CACHE[key] = table[indices] if indices is not None else np.array([], dtype=table.dtype)
    return _GET_ALL_CACHE[key]

def get_all_conjunctive(labels_values, table=vocab):
    """
    :param labels_values: list of (l,v) pairs: [(l1, v1), (l2, v2), (l3, v3)]
    :return: vocab items with the given value for each label
    """
    key = (tuple(labels_values), _table_cache_key(table))
    if key not in _GET_ALL_CONJ_CACHE:
        # Intersect index sets from smallest to largest (early-exit if any is empty)
        idx_sets = []
        for label, value in labels_values:
            idx = _get_label_index(label, table)
            arr = idx.get(str(value))
            if arr is None or len(arr) == 0:
                _GET_ALL_CONJ_CACHE[key] = np.array([], dtype=table.dtype)
                return _GET_ALL_CONJ_CACHE[key]
            idx_sets.append(arr)
        idx_sets.sort(key=len)
        result_indices = idx_sets[0]
        for other in idx_sets[1:]:
            result_indices = np.intersect1d(result_indices, other, assume_unique=True)
            if len(result_indices) == 0:
                break
        _GET_ALL_CONJ_CACHE[key] = table[result_indices]
    return _GET_ALL_CONJ_CACHE[key]


def get_matches_of(row, label, table=vocab):
    """
    :param row: ndarray row. functor vocab item.
    :param label: string. field containing selectional restrictions.
    :param table: ndarray of vocab items.
    :return: all entries in table that match the selectional restrictions of row as given in label.
    """
    value = str(row[label])
    if value == "":
        pass
    else:
        key = (row_signature(row), label, _table_cache_key(table))
        if key not in _GET_MATCHES_OF_CACHE:
            disjuncts = value.split(";")
            if len(disjuncts) == 1:
                result = get_all_conjunctive(conj_list(disjuncts[0]), table)
            else:
                # Union of disjuncts via boolean mask (avoids duplicates)
                mask = np.zeros(len(table), dtype=bool)
                for disjunct in disjuncts:
                    k_vs = conj_list(disjunct)
                    conj_mask = np.ones(len(table), dtype=bool)
                    for lbl, val in k_vs:
                        conj_mask &= (table[lbl] == val)
                    mask |= conj_mask
                result = table[mask]
            if len(_GET_MATCHES_OF_CACHE) >= _RESULT_CACHE_MAX:
                _GET_MATCHES_OF_CACHE.clear()
            _GET_MATCHES_OF_CACHE[key] = result
        return _GET_MATCHES_OF_CACHE[key]


def get_matches_of_conj(rows_labels, table=vocab):
    """
    :param rows_labels: list of (r,l) pairs: [(r1, l1), (r2, l2), (r3, l3)]
    :param table: ndarray of vocab items.
    :return: all entries in table that match the selectional restrictions of all rows as given by labels.
    """
    to_return = table
    for row, label in rows_labels:
        value = str(row[label])
        if value == "":
            pass
        else:
            to_return = get_matches_of(row, label, to_return)
    return to_return


def get_matched_by(row, label, table=vocab):
    """
    :param row: ndarray row. selected vocab item.
    :param label: string. field containing selectional restrictions.
    :param table: ndarray of vocab items.
    :return: all entries in table whose selectional restrictions in label are matched by row.
    """
    key = (row_signature(row), label, _table_cache_key(table))
    if key not in _GET_MATCHED_BY_CACHE:
        idx = _get_label_index(label, table)
        passing_indices = [indices for val, indices in idx.items() if is_match_disj(row, val)]
        if passing_indices:
            result_indices = np.concatenate(passing_indices)
            result = table[result_indices]
        else:
            result = np.array([], dtype=table.dtype)
        if len(_GET_MATCHED_BY_CACHE) >= _RESULT_CACHE_MAX:
            _GET_MATCHED_BY_CACHE.clear()
        _GET_MATCHED_BY_CACHE[key] = result
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
