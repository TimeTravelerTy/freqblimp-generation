import csv
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
_TABLE_LABEL_VALUE_INDEX = {}  # (label, value, table_key) -> np.array of row indices
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
_WARNED_MESSAGES = set()
_HIGH_CARDINALITY_EXACT_MATCH_LABELS = frozenset({"expression", "root", "singularform", "pluralform"})
_RUNTIME_COMPACT_CACHE_VERSION = "v2"
_EXCLUDED_OVERLAY_NOUN_EXPRESSIONS = frozenset({
    "chang",
    "qaeda",
})


def _normalize_expression_key(value):
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _expression_named_entity_variants(expression):
    variants = {_normalize_expression_key(expression)}
    expr = next(iter(variants))
    if " " not in expr:
        if expr.endswith("ies") and len(expr) > 3:
            variants.add(expr[:-3] + "y")
        if expr.endswith("s") and len(expr) > 3:
            variants.add(expr[:-1])
    return variants


@lru_cache(maxsize=1)
def _base_named_entity_expressions():
    expressions = set()
    if not os.path.exists(BASE_VOCAB_PATH):
        return frozenset()
    with open(BASE_VOCAB_PATH, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("properNoun") == "1" or row.get("locale") == "1":
                expression = _normalize_expression_key(row.get("expression", ""))
                if expression:
                    expressions.add(expression)
    return frozenset(expressions)


def _acronym_like_mask(noun_values, expressions):
    noun_values = np.asarray(noun_values)
    expressions = np.asarray(expressions, dtype=str)
    is_noun = noun_values == "1"
    if not np.any(is_noun):
        return np.zeros(len(noun_values), dtype=bool)
    has_vowel = np.zeros(len(expressions), dtype=bool)
    for v in "aeiouAEIOU":
        has_vowel |= np.char.find(expressions, v) >= 0
    return is_noun & ~has_vowel


def _overlay_common_noun_noise_mask(noun_values, expressions, proper_values=None, locale_values=None):
    noun_values = np.asarray(noun_values)
    expressions = np.asarray(expressions, dtype=str)
    is_noun = noun_values == "1"
    if not np.any(is_noun):
        return np.zeros(len(noun_values), dtype=bool)
    proper_values = np.asarray(proper_values if proper_values is not None else np.repeat("", len(noun_values)), dtype=str)
    locale_values = np.asarray(locale_values if locale_values is not None else np.repeat("", len(noun_values)), dtype=str)
    normalized = np.char.lower(np.char.strip(expressions))
    one_token = np.char.find(normalized, " ") < 0
    short_alpha = one_token & np.char.isalpha(normalized) & (np.char.str_len(normalized) <= 3)
    explicit_blocklist = np.isin(normalized, list(_EXCLUDED_OVERLAY_NOUN_EXPRESSIONS))
    same_as_named_entity = np.zeros(len(noun_values), dtype=bool)
    noun_indices = np.flatnonzero(is_noun)
    named_entities = _base_named_entity_expressions()
    if named_entities:
        noun_exprs = normalized[noun_indices].tolist()
        same_as_named_entity[noun_indices] = np.fromiter(
            (bool(_expression_named_entity_variants(expr) & named_entities) for expr in noun_exprs),
            dtype=bool,
            count=len(noun_indices),
        )
    is_locale_or_proper = (proper_values == "1") | (locale_values == "1")
    return is_noun & (short_alpha | explicit_blocklist | same_as_named_entity | is_locale_or_proper)


def _overlay_row_allowed(values, field_positions):
    if values[field_positions["noun"]] != "1":
        return True
    expression = _normalize_expression_key(values[field_positions["expression"]])
    if not expression:
        return False
    if values[field_positions.get("properNoun", -1)] == "1":
        return False
    if values[field_positions.get("locale", -1)] == "1":
        return False
    if expression in _EXCLUDED_OVERLAY_NOUN_EXPRESSIONS:
        return False
    if " " not in expression and expression.isalpha() and len(expression) <= 3:
        return False
    if _expression_named_entity_variants(expression) & _base_named_entity_expressions():
        return False
    if not any(ch in "aeiou" for ch in expression):
        return False
    return True


class FilteredTable:
    def __init__(self, source, exclude_acronym_nouns=False):
        self.source = source
        self.exclude_acronym_nouns = exclude_acronym_nouns
        self.dtype = source.dtype
        self.shape = None
        self.strides = None
        self._indices = None
        self._keep_mask = None

    def keep_mask(self):
        if self._keep_mask is None:
            keep = np.asarray(self.source["OOV_inductive_biases"] != "1", dtype=bool)
            if self.exclude_acronym_nouns:
                keep &= ~_overlay_common_noun_noise_mask(
                    self.source["noun"],
                    self.source["expression"],
                    proper_values=self.source["properNoun"],
                    locale_values=self.source["locale"],
                )
            self._keep_mask = keep
        return self._keep_mask

    def filter_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if len(indices) == 0:
            return indices
        keep_mask = self.keep_mask()
        return indices[keep_mask[indices]]

    def resolve_indices(self):
        if self._indices is None:
            self._indices = np.flatnonzero(self.keep_mask())
            self.shape = (len(self._indices),)
        return self._indices

    def __len__(self):
        return len(self.resolve_indices())

    def __iter__(self):
        for idx in self.resolve_indices():
            yield self.source[int(idx)].copy()

    def __getitem__(self, item):
        indices = self.resolve_indices()
        if isinstance(item, str):
            return self.source[item][indices]
        selected = indices[item]
        if np.isscalar(selected):
            return self.source[int(selected)].copy()
        return IndexedTable(self.source, selected)

    def __array__(self, dtype=None, copy=None):
        target = dtype if dtype is not None else data_type
        array = np.asarray(self.source[self.resolve_indices()], dtype=target)
        if copy:
            return array.copy()
        return array

    def __array_function__(self, func, types, args, kwargs):
        return _table_array_function_dispatch(func, args, kwargs)


class IndexedTable:
    def __init__(self, source, indices):
        self.source = source
        self.indices = np.asarray(indices, dtype=np.int64)
        self.dtype = source.dtype
        self.shape = (len(self.indices),)
        self.strides = None

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        for idx in self.indices:
            yield self.source[int(idx)].copy()

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.source[item][self.indices]
        if isinstance(item, tuple):
            raise TypeError("IndexedTable does not support tuple indexing")
        selected = self.indices[item]
        if np.isscalar(selected):
            return self.source[int(selected)].copy()
        return IndexedTable(self.source, selected)

    def __array__(self, dtype=None, copy=None):
        target = dtype if dtype is not None else data_type
        array = np.asarray(self.source[self.indices], dtype=target)
        if copy:
            return array.copy()
        return array

    def __array_function__(self, func, types, args, kwargs):
        return _table_array_function_dispatch(func, args, kwargs)


class ConcatTable:
    def __init__(self, parts):
        self.parts = tuple(part for part in parts if len(part) > 0)
        self.dtype = self.parts[0].dtype if self.parts else data_type
        self.shape = (sum(len(part) for part in self.parts),)
        self.strides = None

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for part in self.parts:
            for row in part:
                yield row

    def __getitem__(self, item):
        if isinstance(item, str):
            arrays = [part[item] for part in self.parts]
            if not arrays:
                return np.array([], dtype=self.dtype[item])
            if len(arrays) == 1:
                return arrays[0]
            return np.concatenate(arrays)
        if isinstance(item, tuple):
            raise TypeError("ConcatTable does not support tuple indexing")
        if isinstance(item, slice):
            return np.array(list(self)[item], dtype=self.dtype)
        if isinstance(item, np.ndarray):
            if item.dtype == bool:
                # Boolean mask: split across parts by offset
                result_parts = []
                offset = 0
                for part in self.parts:
                    n = len(part)
                    sub_mask = item[offset:offset + n]
                    if np.any(sub_mask):
                        result_parts.append(part[sub_mask])
                    offset += n
                return _concat_query_results(result_parts, self.dtype) if result_parts else np.array([], dtype=self.dtype)
            # Integer array indexing — per-part scatter (no full materialisation).
            # Sort indices to enable O(log N) range queries per part, then restore
            # original order via inverse permutation.
            item = np.asarray(item, dtype=np.intp)
            if len(item) == 0:
                return np.array([], dtype=self.dtype)
            sort_perm = np.argsort(item, kind="stable")
            sorted_item = item[sort_perm]
            offset = 0
            result_parts = []
            for part in self.parts:
                n = len(part)
                lo = int(np.searchsorted(sorted_item, offset))
                hi = int(np.searchsorted(sorted_item, offset + n))
                if lo < hi:
                    local_indices = sorted_item[lo:hi] - offset
                    result_parts.append(np.asarray(part[local_indices], dtype=self.dtype))
                offset += n
            if not result_parts:
                return np.array([], dtype=self.dtype)
            sorted_result = np.concatenate(result_parts)
            # Restore caller's original index order
            inv_perm = np.empty(len(sort_perm), dtype=np.intp)
            inv_perm[sort_perm] = np.arange(len(sort_perm), dtype=np.intp)
            return sorted_result[inv_perm]
        position = int(item)
        if position < 0:
            position += len(self)
        for part in self.parts:
            if position < len(part):
                return part[position]
            position -= len(part)
        raise IndexError(position)

    def __array__(self, dtype=None, copy=None):
        target = dtype if dtype is not None else data_type
        arrays = [np.asarray(part, dtype=target) for part in self.parts]
        if not arrays:
            result = np.array([], dtype=target)
        elif len(arrays) == 1:
            result = arrays[0]
        else:
            result = np.concatenate(arrays)
        if copy:
            return result.copy()
        return result

    def __array_function__(self, func, types, args, kwargs):
        return _table_array_function_dispatch(func, args, kwargs)


def _table_array_function_dispatch(func, args, kwargs):
    """Route numpy set operations on table objects to index-based implementations.

    Called from __array_function__ on FilteredTable, IndexedTable, ConcatTable.
    Returns NotImplemented for functions we don't handle so numpy falls back.
    """
    if func is np.intersect1d:
        ar1, ar2 = args[0], args[1]
        if kwargs.get("return_indices", False):
            return NotImplemented
        return table_intersect1d(ar1, ar2)
    if func is np.union1d:
        ar1, ar2 = args[0], args[1]
        return table_union1d(ar1, ar2)
    if func is np.setdiff1d:
        ar1, ar2 = args[0], args[1]
        return table_setdiff1d(ar1, ar2)
    # For all other numpy functions, materialise table args to data_type and delegate.
    _TABLE_TYPES = (FilteredTable, IndexedTable, ConcatTable)
    new_args = tuple(np.asarray(a, dtype=data_type) if isinstance(a, _TABLE_TYPES) else a for a in args)
    new_kwargs = {k: np.asarray(v, dtype=data_type) if isinstance(v, _TABLE_TYPES) else v for k, v in kwargs.items()}
    return func(*new_args, **new_kwargs)


def _warn_once(key, message):
    if key in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(key)
    print(message, flush=True)


def _normalize_table(table):
    if hasattr(table, "resolve"):
        return table.resolve()
    return table


def clear_query_caches():
    global _OVERLAY_METADATA_REGISTRY, _OVERLAY_METADATA_REGISTRY_PATH
    global _EXPRESSION_ZIPF_REGISTRY, _EXPRESSION_ZIPF_REGISTRY_PATH
    _GET_ALL_CACHE.clear()
    _GET_ALL_CONJ_CACHE.clear()
    _GET_MATCHES_OF_CACHE.clear()
    _GET_MATCHED_BY_CACHE.clear()
    _TABLE_LABEL_INDEX.clear()
    _TABLE_LABEL_VALUE_INDEX.clear()
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
    table = _normalize_table(table)
    tkey = (label, _table_cache_key(table))
    if tkey not in _TABLE_LABEL_INDEX:
        if isinstance(table, FilteredTable):
            base_idx = _get_label_index(label, table.source)
            filtered = {}
            for value, indices in base_idx.items():
                kept = table.filter_indices(indices)
                if len(kept) > 0:
                    filtered[str(value)] = kept
            _TABLE_LABEL_INDEX[tkey] = filtered
        elif isinstance(table, IndexedTable):
            base_idx = _get_label_index(label, table.source)
            filtered = {}
            for value, indices in base_idx.items():
                kept = np.intersect1d(indices, table.indices, assume_unique=False)
                if len(kept) > 0:
                    filtered[str(value)] = kept
            _TABLE_LABEL_INDEX[tkey] = filtered
        else:
            col = table[label]
            if len(col) > 100000:
                grouped = defaultdict(list)
                for idx, value in enumerate(col):
                    grouped[str(value)].append(idx)
                _TABLE_LABEL_INDEX[tkey] = {
                    value: np.asarray(indices, dtype=np.int64)
                    for value, indices in grouped.items()
                }
            else:
                unique_vals, inverse = np.unique(col, return_inverse=True)
                sort_order = np.argsort(inverse, kind="stable")
                sorted_codes = inverse[sort_order]
                splits = np.where(np.diff(sorted_codes))[0] + 1
                groups = np.split(sort_order, splits)
                _TABLE_LABEL_INDEX[tkey] = {str(unique_vals[i]): groups[i] for i in range(len(unique_vals))}
    return _TABLE_LABEL_INDEX[tkey]


def _should_scan_exact_match(label, table):
    table = _normalize_table(table)
    if label not in _HIGH_CARDINALITY_EXACT_MATCH_LABELS:
        return False
    if isinstance(table, (FilteredTable, IndexedTable)):
        return True
    return len(table) > 100000


def _get_label_value_indices(label, value, table):
    table = _normalize_table(table)
    value = str(value)
    key = (label, value, _table_cache_key(table))
    if key in _TABLE_LABEL_VALUE_INDEX:
        return _TABLE_LABEL_VALUE_INDEX[key]
    empty = np.array([], dtype=np.int64)
    if isinstance(table, FilteredTable):
        result = table.filter_indices(_get_label_value_indices(label, value, table.source))
    elif isinstance(table, IndexedTable):
        if len(table.indices) == 0:
            result = empty
        else:
            result = table.indices[table.source[label][table.indices] == value]
    elif _should_scan_exact_match(label, table):
        result = np.flatnonzero(table[label] == value).astype(np.int64, copy=False)
    else:
        result = np.asarray(_get_label_index(label, table).get(value, empty), dtype=np.int64)
    _TABLE_LABEL_VALUE_INDEX[key] = result
    return result


def _env_flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _decode_entry(entry):
    entry = entry.copy()
    entry[0] = re.sub("!", "'", entry[0])
    return entry


def widen_expression_field(row, min_width=512):
    dtype = getattr(row, "dtype", None)
    if dtype is None or not getattr(dtype, "names", None) or "expression" not in dtype.names:
        return row.copy()
    expr_dtype = dtype.fields["expression"][0]
    if expr_dtype.kind != "U":
        return row.copy()
    current_width = expr_dtype.itemsize // 4
    if current_width >= min_width:
        return row.copy()
    widened_dtype = []
    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        if name == "expression":
            widened_dtype.append((name, "U%d" % max(min_width, current_width)))
        else:
            widened_dtype.append((name, field_dtype))
    widened = np.empty((), dtype=np.dtype(widened_dtype))
    for name in dtype.names:
        widened[name] = row[name]
    return widened[()]


def _load_vocab_csv(path, mmap_mode=None):
    return _load_vocab_csv_runtime(path, mmap_mode=mmap_mode)


def _load_vocab_csv_runtime(path, mmap_mode=None, exclude_acronym_nouns=False):
    if not path or not os.path.exists(path):
        return np.array([], dtype=data_type)
    compact_cache = path + ".runtime.compact.%s.npy" % _RUNTIME_COMPACT_CACHE_VERSION
    if os.path.exists(compact_cache) and os.path.getmtime(compact_cache) >= os.path.getmtime(path):
        try:
            table = np.load(compact_cache, allow_pickle=False, mmap_mode=mmap_mode)
            expected_names = tuple(field for field, _dtype in data_type)
            if getattr(getattr(table, "dtype", None), "names", ()) == expected_names:
                return table
        except Exception:
            pass
    try:
        _write_compact_vocab_npy(path, compact_cache, exclude_acronym_nouns=exclude_acronym_nouns)
    except Exception:
        legacy_cache = path + ".npy"
        if os.path.exists(legacy_cache) and os.path.getmtime(legacy_cache) >= os.path.getmtime(path):
            try:
                return np.load(legacy_cache, allow_pickle=False, mmap_mode=mmap_mode)
            except Exception:
                pass
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=data_type)
        if getattr(table, "shape", ()) == ():
            table = np.array([table], dtype=data_type)
        table = table.copy()
        table["expression"] = np.char.replace(table["expression"], "!", "'")
        table = _filter_runtime_vocab(table)
        if exclude_acronym_nouns:
            table = _filter_overlay_acronym_nouns(table)
        return table
    if os.path.exists(compact_cache):
        try:
            return np.load(compact_cache, allow_pickle=False, mmap_mode=mmap_mode)
        except Exception:
            pass
    return np.array([], dtype=data_type)


def _row_allowed(values, field_positions, exclude_acronym_nouns=False):
    if values[field_positions["OOV_inductive_biases"]] == "1":
        return False
    if exclude_acronym_nouns and not _overlay_row_allowed(values, field_positions):
        return False
    return True


def _infer_compact_dtype(path, exclude_acronym_nouns=False):
    canonical_fields = [field for field, _dtype in data_type]
    field_positions = {field: idx for idx, field in enumerate(canonical_fields)}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header row; runtime code uses canonical data_type order
        max_widths = {field: 1 for field in canonical_fields}
        row_count = 0
        for row in reader:
            values = [
                (row[idx] if idx < len(row) else "").replace("!", "'") if field == "expression" else (row[idx] if idx < len(row) else "")
                for idx, field in enumerate(canonical_fields)
            ]
            if not _row_allowed(values, field_positions, exclude_acronym_nouns=exclude_acronym_nouns):
                continue
            row_count += 1
            for idx, field in enumerate(canonical_fields):
                value = values[idx]
                width = len(str(value))
                if width > max_widths[field]:
                    max_widths[field] = width
    dtype = np.dtype([(field, "U%d" % max(1, max_widths[field])) for field in canonical_fields])
    return canonical_fields, dtype, row_count


def _write_compact_vocab_npy(csv_path, npy_path, exclude_acronym_nouns=False):
    fieldnames, compact_dtype, row_count = _infer_compact_dtype(csv_path, exclude_acronym_nouns=exclude_acronym_nouns)
    field_positions = {field: idx for idx, field in enumerate(fieldnames)}
    tmp_path = npy_path + ".tmp.npy"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    table = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=compact_dtype, shape=(row_count,))
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        out_idx = 0
        for idx, row in enumerate(reader):
            values = [
                (row[field_idx] if field_idx < len(row) else "").replace("!", "'") if field == "expression" else (row[field_idx] if field_idx < len(row) else "")
                for field_idx, field in enumerate(fieldnames)
            ]
            if not _row_allowed(values, field_positions, exclude_acronym_nouns=exclude_acronym_nouns):
                continue
            for field_idx, field in enumerate(fieldnames):
                table[field][out_idx] = values[field_idx]
            out_idx += 1
    table.flush()
    del table
    os.replace(tmp_path, npy_path)


def _max_load_bytes(env_var, default_bytes):
    raw = os.environ.get(env_var, os.environ.get("FREQBLIMP_MAX_JSON_LOAD_BYTES"))
    if raw is None or str(raw).strip() == "":
        return int(default_bytes)
    value = int(raw)
    if value <= 0:
        return None
    return value


def _skip_large_json_load(path, env_var, default_bytes, label):
    if _env_flag("FREQBLIMP_ALLOW_LARGE_JSON_LOADS", default=False):
        return False
    if not path or not os.path.exists(path):
        return False
    limit = _max_load_bytes(env_var, default_bytes)
    if limit is None:
        return False
    size = os.path.getsize(path)
    if size <= limit:
        return False
    _warn_once(
        (label, path),
        "[freq-blimp] skipping %s load for %s (%.1f MiB > %.1f MiB limit); falling back to lazy lookups"
        % (label, path, size / (1024 * 1024), limit / (1024 * 1024)),
    )
    return True


def _load_overlay_manifest(path):
    if not path or not os.path.exists(path):
        return {}
    if _skip_large_json_load(
        path,
        "FREQBLIMP_MAX_MANIFEST_LOAD_BYTES",
        default_bytes=64 * 1024 * 1024,
        label="overlay manifest",
    ):
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
    if _skip_large_json_load(
        path,
        "FREQBLIMP_MAX_FREQUENCY_CACHE_LOAD_BYTES",
        default_bytes=64 * 1024 * 1024,
        label="frequency cache",
    ):
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


def _overlay_acronym_noun_mask(table):
    """Rows that look like lowercased acronyms (no standard vowels, e.g. 'nlp', 'mvp', 'psf')."""
    return _acronym_like_mask(table["noun"], table["expression"])


def _filter_overlay_acronym_nouns(table):
    return table[~_overlay_acronym_noun_mask(table)]


def _is_composite_table(table):
    return isinstance(table, (tuple, ConcatTable))


def _iter_tables(table):
    if isinstance(table, ConcatTable):
        return table.parts
    if isinstance(table, tuple):
        return table
    return (table,)


def _concat_query_results(parts, dtype):
    nonempty = [part for part in parts if len(part) > 0]
    if not nonempty:
        return np.array([], dtype=dtype)
    if len(nonempty) == 1:
        return nonempty[0]
    return ConcatTable(nonempty)


def _table_slice_result(table, indices):
    if indices is None or len(indices) == 0:
        return np.array([], dtype=table.dtype)
    if isinstance(table, np.memmap):
        return IndexedTable(table, indices)
    return table[indices]


def _parts_by_source(table):
    table = _normalize_table(table)
    if isinstance(table, ConcatTable):
        merged = {}
        for part in table.parts:
            part_map = _parts_by_source(part)
            if part_map is None:
                return None
            for key, (source, indices) in part_map.items():
                if key in merged:
                    merged[key] = (source, np.union1d(merged[key][1], indices))
                else:
                    merged[key] = (source, np.asarray(indices, dtype=np.int64))
        return merged
    if isinstance(table, IndexedTable):
        return {id(table.source): (table.source, np.asarray(table.indices, dtype=np.int64))}
    if isinstance(table, FilteredTable):
        indices = np.asarray(table.resolve_indices(), dtype=np.int64)
        return {id(table.source): (table.source, indices)}
    if isinstance(table, np.ndarray):
        if len(table) == 0:
            return {}
        return {id(table): (table, np.arange(len(table), dtype=np.int64))}
    return None


def _table_from_parts_map(parts_map, dtype=data_type):
    if not parts_map:
        return np.array([], dtype=dtype)
    parts = [IndexedTable(source, indices) for _key, (source, indices) in sorted(parts_map.items(), key=lambda item: item[0]) if len(indices) > 0]
    return _concat_query_results(parts, dtype)


def table_union1d(left, right):
    left_parts = _parts_by_source(left)
    right_parts = _parts_by_source(right)
    if left_parts is None or right_parts is None:
        return np.union1d(np.asarray(_normalize_table(left)), np.asarray(_normalize_table(right)))
    merged = dict(left_parts)
    for key, (source, indices) in right_parts.items():
        if key in merged:
            merged[key] = (source, np.union1d(merged[key][1], indices))
        else:
            merged[key] = (source, np.asarray(indices, dtype=np.int64))
    dtype = getattr(_normalize_table(left), "dtype", getattr(_normalize_table(right), "dtype", data_type))
    return _table_from_parts_map(merged, dtype=dtype)


def table_intersect1d(left, right):
    left_parts = _parts_by_source(left)
    right_parts = _parts_by_source(right)
    if left_parts is None or right_parts is None:
        return np.intersect1d(np.asarray(_normalize_table(left)), np.asarray(_normalize_table(right)))
    merged = {}
    for key, (source, indices) in left_parts.items():
        if key not in right_parts:
            continue
        kept = np.intersect1d(indices, right_parts[key][1], assume_unique=False)
        if len(kept) > 0:
            merged[key] = (source, kept)
    dtype = getattr(_normalize_table(left), "dtype", getattr(_normalize_table(right), "dtype", data_type))
    return _table_from_parts_map(merged, dtype=dtype)


def table_setdiff1d(left, right):
    left_parts = _parts_by_source(left)
    right_parts = _parts_by_source(right)
    if left_parts is None or right_parts is None:
        return np.setdiff1d(np.asarray(_normalize_table(left)), np.asarray(_normalize_table(right)))
    merged = {}
    for key, (source, indices) in left_parts.items():
        if key in right_parts:
            kept = np.setdiff1d(indices, right_parts[key][1], assume_unique=False)
        else:
            kept = indices
        if len(kept) > 0:
            merged[key] = (source, kept)
    dtype = getattr(_normalize_table(left), "dtype", data_type)
    return _table_from_parts_map(merged, dtype=dtype)


def _build_runtime_vocab():
    base_vocab = _load_vocab_csv_runtime(BASE_VOCAB_PATH, mmap_mode="r", exclude_acronym_nouns=False)
    if _env_flag("FREQBLIMP_USE_OVERLAY", default=False):
        overlay_path = os.environ.get("FREQBLIMP_VOCAB_OVERLAY", DEFAULT_OVERLAY_PATH)
        overlay_vocab = _load_vocab_csv_runtime(overlay_path, mmap_mode="r", exclude_acronym_nouns=True)
        if len(overlay_vocab) > 0:
            return ConcatTable([base_vocab, overlay_vocab])
    return base_vocab


def _filter_runtime_vocab(table):
    return table[table["OOV_inductive_biases"] != "1"]


# Build once at import time. Overlay mode uses an on-disk memmap cache so the
# runtime vocabulary does not have to exist as a giant in-memory ndarray.
vocab = _build_runtime_vocab()

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
    table = _normalize_table(table)
    key = _table_cache_key(table)
    if key not in _TABLE_ZIPF_EXPRESSION_CACHE:
        if _is_composite_table(table):
            parts = _iter_tables(table)
            _TABLE_ZIPF_EXPRESSION_CACHE[key] = np.concatenate(
                [get_table_zipf_expression(part) for part in parts if len(part) > 0]
            ) if any(len(part) > 0 for part in parts) else np.array([], dtype=np.float32)
            return _TABLE_ZIPF_EXPRESSION_CACHE[key]
        expressions = np.asarray(table["expression"], dtype=str)
        unique_expressions, inverse = np.unique(expressions, return_inverse=True)
        unique_zipf = np.array(
            [_zipf_for_cached_expression(expression) for expression in unique_expressions],
            dtype=np.float32,
        )
        _TABLE_ZIPF_EXPRESSION_CACHE[key] = unique_zipf[inverse]
    return _TABLE_ZIPF_EXPRESSION_CACHE[key]


def _table_cache_key(table):
    table = _normalize_table(table)
    if isinstance(table, ConcatTable):
        return ("concat", tuple(_table_cache_key(part) for part in table.parts))
    if isinstance(table, tuple):
        return tuple(_table_cache_key(part) for part in table)
    if isinstance(table, FilteredTable):
        return ("filtered", _table_cache_key(table.source), bool(table.exclude_acronym_nouns))
    if isinstance(table, IndexedTable):
        interface = getattr(table.indices, "__array_interface__", None)
        data_ptr = None
        if interface and interface.get("data"):
            data_ptr = interface["data"][0]
        return (
            "indexed",
            _table_cache_key(table.source),
            data_ptr,
            len(table.indices),
        )
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
    table = _normalize_table(table)
    key = (label, value, _table_cache_key(table))
    if key not in _GET_ALL_CACHE:
        if _is_composite_table(table):
            parts = [get_all(label, value, part) for part in _iter_tables(table)]
            dtype = next((part.dtype for part in _iter_tables(table)), data_type)
            _GET_ALL_CACHE[key] = _concat_query_results(parts, dtype)
        elif isinstance(table, (FilteredTable, IndexedTable)):
            indices = _get_label_value_indices(label, value, table)
            _GET_ALL_CACHE[key] = IndexedTable(table.source, indices) if len(indices) > 0 else np.array([], dtype=table.dtype)
        else:
            _GET_ALL_CACHE[key] = _table_slice_result(table, _get_label_value_indices(label, value, table))
    return _GET_ALL_CACHE[key]

def get_all_conjunctive(labels_values, table=vocab):
    """
    :param labels_values: list of (l,v) pairs: [(l1, v1), (l2, v2), (l3, v3)]
    :return: vocab items with the given value for each label
    """
    table = _normalize_table(table)
    key = (tuple(labels_values), _table_cache_key(table))
    if key not in _GET_ALL_CONJ_CACHE:
        if _is_composite_table(table):
            parts = [get_all_conjunctive(labels_values, part) for part in _iter_tables(table)]
            dtype = next((part.dtype for part in _iter_tables(table)), data_type)
            _GET_ALL_CONJ_CACHE[key] = _concat_query_results(parts, dtype)
            return _GET_ALL_CONJ_CACHE[key]
        # Intersect index sets from smallest to largest (early-exit if any is empty)
        idx_sets = []
        for label, value in labels_values:
            arr = _get_label_value_indices(label, value, table)
            if len(arr) == 0:
                _GET_ALL_CONJ_CACHE[key] = np.array([], dtype=table.dtype)
                return _GET_ALL_CONJ_CACHE[key]
            idx_sets.append(arr)
        idx_sets.sort(key=len)
        result_indices = idx_sets[0]
        for other in idx_sets[1:]:
            result_indices = np.intersect1d(result_indices, other, assume_unique=True)
            if len(result_indices) == 0:
                break
        if isinstance(table, (FilteredTable, IndexedTable)):
            _GET_ALL_CONJ_CACHE[key] = IndexedTable(table.source, result_indices)
        else:
            _GET_ALL_CONJ_CACHE[key] = _table_slice_result(table, result_indices)
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
        table = _normalize_table(table)
        key = (row_signature(row), label, _table_cache_key(table))
        if key not in _GET_MATCHES_OF_CACHE:
            if _is_composite_table(table):
                parts = [get_matches_of(row, label, part) for part in _iter_tables(table)]
                dtype = next((part.dtype for part in _iter_tables(table)), data_type)
                _GET_MATCHES_OF_CACHE[key] = _concat_query_results(parts, dtype)
                return _GET_MATCHES_OF_CACHE[key]
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
    to_return = _normalize_table(table)
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
    table = _normalize_table(table)
    key = (row_signature(row), label, _table_cache_key(table))
    if key not in _GET_MATCHED_BY_CACHE:
        if _is_composite_table(table):
            parts = [get_matched_by(row, label, part) for part in _iter_tables(table)]
            dtype = next((part.dtype for part in _iter_tables(table)), data_type)
            result = _concat_query_results(parts, dtype)
        else:
            idx = _get_label_index(label, table)
            passing_indices = [indices for val, indices in idx.items() if is_match_disj(row, val)]
            if passing_indices:
                result_indices = np.concatenate(passing_indices)
                if isinstance(table, (FilteredTable, IndexedTable)):
                    result = IndexedTable(table.source, result_indices)
                else:
                    result = _table_slice_result(table, result_indices)
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
