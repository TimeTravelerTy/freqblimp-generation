from functools import reduce

import numpy as np

from utils.randomize import *
from utils.vocab_table import *


_BLOCKED_NOUN_EXPRESSIONS = (
    "asshole",
    "bitch",
    "cock",
    "cunt",
    "dick",
    "dildo",
    "demon",
    "demons",
    "femicide",
    "femicides",
    "fag",
    "faggot",
    "fuck",
    "idiot",
    "idiots",
    "instagram",
    "ipod",
    "ipods",
    "motherfucker",
    "nazi",
    "nazis",
    "nato",
    "penis",
    "pussy",
    "raper",
    "rapers",
    "scholasticism",
    "scholasticisms",
    "slave",
    "slaves",
    "slut",
    "tit",
    "tits",
    "twat",
    "vagina",
    "whore",
    "butt",
)

_BLOCKED_VERB_EXPRESSIONS = (
    "according",
)


def _exclude_bad_noun_expressions(table):
    blocked_rows = []
    for expression in _BLOCKED_NOUN_EXPRESSIONS:
        matches = get_all("expression", expression, table)
        if len(matches) > 0:
            blocked_rows.append(matches)
    if not blocked_rows:
        return table
    blocked = reduce(table_union1d, blocked_rows)
    return table_setdiff1d(table, blocked)


def _exclude_bad_verb_expressions(table):
    blocked_rows = []
    for expression in _BLOCKED_VERB_EXPRESSIONS:
        matches = get_all("expression", expression, table)
        if len(matches) > 0:
            blocked_rows.append(matches)
    if not blocked_rows:
        return table
    blocked = reduce(table_union1d, blocked_rows)
    return table_setdiff1d(table, blocked)


class LazyVocabSet:
    def __init__(self, name, builder):
        self._name = name
        self._builder = builder
        self._value = None
        import utils.vocab_table as _vt
        _vt._LAZY_REGISTRY.append(self)

    def resolve(self):
        if self._value is None:
            self._value = self._builder()
        return self._value

    def __array__(self, dtype=None, copy=None):
        array = np.asarray(self.resolve(), dtype=dtype)
        if copy:
            return array.copy()
        return array

    def __array_function__(self, func, types, args, kwargs):
        return func(*_resolve_lazy(args), **_resolve_lazy(kwargs))

    def __len__(self):
        return len(self.resolve())

    def __iter__(self):
        return iter(self.resolve())

    def __getitem__(self, item):
        return self.resolve()[item]

    def __getattr__(self, name):
        return getattr(self.resolve(), name)

    def __repr__(self):
        status = "ready" if self._value is not None else "lazy"
        return f"<LazyVocabSet {self._name} {status}>"


def _lazy(name, builder):
    return LazyVocabSet(name, builder)


def _resolve_lazy(value):
    if isinstance(value, LazyVocabSet):
        return value.resolve()
    if isinstance(value, tuple):
        return tuple(_resolve_lazy(item) for item in value)
    if isinstance(value, list):
        return [_resolve_lazy(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_lazy(item) for key, item in value.items()}
    return value


def _verbs_matching_argument_sets(subject_rows, object_rows, verb_space):
    subject_rows = _resolve_lazy(subject_rows)
    object_rows = _resolve_lazy(object_rows)
    verb_space = _resolve_lazy(verb_space)
    kept_indices = []
    for idx, verb in enumerate(verb_space):
        if len(get_matches_of(verb, "arg_1", subject_rows)) == 0:
            continue
        if len(get_matches_of(verb, "arg_2", object_rows)) == 0:
            continue
        kept_indices.append(idx)
    if not kept_indices:
        return np.array([], dtype=verb_space.dtype)
    return verb_space[np.asarray(kept_indices, dtype=np.int64)]


# NOUNS
all_nouns = _lazy(
    "all_nouns",
    lambda: _exclude_bad_noun_expressions(
        table_setdiff1d(get_all("category", "N"), get_all("locale", "1"))
    ),
)
all_singular_nouns = _lazy("all_singular_nouns", lambda: get_all("sg", "1", all_nouns))
all_singular_count_nouns = _lazy("all_singular_count_nouns", lambda: get_all("mass", "0", all_singular_nouns))
all_animate_nouns = _lazy("all_animate_nouns", lambda: get_all("animate", "1", all_nouns))
all_inanimate_nouns = _lazy("all_inanimate_nouns", lambda: get_all("animate", "0", all_nouns))
all_documents = _lazy("all_documents", lambda: get_all_conjunctive([("category", "N"), ("document", "1")]))
all_gendered_nouns = _lazy(
    "all_gendered_nouns",
    lambda: table_union1d(get_all("gender", "m"), get_all("gender", "f")),
)
all_singular_neuter_animate_nouns = _lazy(
    "all_singular_neuter_animate_nouns",
    lambda: get_all_conjunctive([("category", "N"), ("sg", "1"), ("animate", "1"), ("gender", "n")]),
)
all_plural_nouns = _lazy("all_plural_nouns", lambda: get_all_conjunctive([("category", "N"), ("pl", "1")]))
all_plural_animate_nouns = _lazy(
    "all_plural_animate_nouns",
    lambda: table_intersect1d(all_animate_nouns, all_plural_nouns),
)
all_common_nouns = _lazy(
    "all_common_nouns",
    lambda: _exclude_bad_noun_expressions(
        table_setdiff1d(get_all_conjunctive([("category", "N"), ("properNoun", "0")]), get_all("locale", "1"))
    ),
)
all_relational_nouns = _lazy("all_relational_nouns", lambda: get_all("category", "N/NP"))
all_nominals = _lazy(
    "all_nominals",
    lambda: _exclude_bad_noun_expressions(
        table_setdiff1d(get_all("noun", "1"), get_all("locale", "1"))
    ),
)
all_relational_poss_nouns = _lazy("all_relational_poss_nouns", lambda: get_all("category", "N\\NP[poss]"))
all_proper_names = _lazy("all_proper_names", lambda: get_all("properNoun", "1"))

# VERBS
all_verbs = _lazy("all_verbs", lambda: _exclude_bad_verb_expressions(get_all("verb", "1")))
all_transitive_verbs = _lazy("all_transitive_verbs", lambda: get_all("category", "(S\\NP)/NP"))
all_intransitive_verbs = _lazy("all_intransitive_verbs", lambda: get_all("category", "S\\NP"))
all_non_recursive_verbs = _lazy(
    "all_non_recursive_verbs",
    lambda: table_union1d(all_transitive_verbs, all_intransitive_verbs),
)
all_finite_verbs = _lazy("all_finite_verbs", lambda: get_all("finite", "1", all_verbs))
all_non_finite_verbs = _lazy("all_non_finite_verbs", lambda: get_all("finite", "0", all_verbs))
all_ing_verbs = _lazy("all_ing_verbs", lambda: get_all("ing", "1", all_verbs))
all_en_verbs = _lazy("all_en_verbs", lambda: get_all("en", "1", all_verbs))
all_bare_verbs = _lazy("all_bare_verbs", lambda: get_all("bare", "1", all_verbs))
all_anim_anim_verbs = _lazy(
    "all_anim_anim_verbs",
    lambda: _verbs_matching_argument_sets(all_animate_nouns, all_animate_nouns, all_transitive_verbs),
)
all_doc_doc_verbs = _lazy(
    "all_doc_doc_verbs",
    lambda: _verbs_matching_argument_sets(all_documents, all_documents, all_transitive_verbs),
)
all_refl_nonverbal_predicates = _lazy(
    "all_refl_nonverbal_predicates",
    lambda: (lambda predicates: predicates[predicates["arg_1"] == predicates["arg_2"]])(get_all("category_2", "Pred")),
)
all_refl_preds = _lazy("all_refl_preds", lambda: reduce(table_union1d, (all_anim_anim_verbs, all_doc_doc_verbs)))
all_non_plural_transitive_verbs = _lazy(
    "all_non_plural_transitive_verbs",
    lambda: (lambda arg_1: all_transitive_verbs[(np.char.find(arg_1, "sg=0") < 0) & (np.char.find(arg_1, "pl=1") < 0)])(
        all_transitive_verbs["arg_1"]
    ),
)
all_strictly_plural_verbs = _lazy(
    "all_strictly_plural_verbs",
    lambda: get_all_conjunctive([("pres", "1"), ("3sg", "0")], all_verbs),
)
all_strictly_singular_verbs = _lazy(
    "all_strictly_singular_verbs",
    lambda: get_all_conjunctive([("pres", "1"), ("3sg", "1")], all_verbs),
)
all_strictly_plural_transitive_verbs = _lazy(
    "all_strictly_plural_transitive_verbs",
    lambda: table_intersect1d(all_strictly_plural_verbs, all_transitive_verbs),
)
all_strictly_singular_transitive_verbs = _lazy(
    "all_strictly_singular_transitive_verbs",
    lambda: table_intersect1d(all_strictly_singular_verbs, all_transitive_verbs),
)
all_possibly_plural_verbs = _lazy(
    "all_possibly_plural_verbs",
    lambda: table_setdiff1d(all_verbs, all_strictly_singular_verbs),
)
all_possibly_singular_verbs = _lazy(
    "all_possibly_singular_verbs",
    lambda: table_setdiff1d(all_verbs, all_strictly_plural_verbs),
)
all_non_finite_transitive_verbs = _lazy(
    "all_non_finite_transitive_verbs",
    lambda: table_intersect1d(all_non_finite_verbs, all_transitive_verbs),
)
all_non_finite_intransitive_verbs = _lazy(
    "all_non_finite_intransitive_verbs",
    lambda: get_all("finite", "0", all_intransitive_verbs),
)
all_modals_auxs = _lazy("all_modals_auxs", lambda: get_all("category", "(S\\NP)/(S[bare]\\NP)"))
all_modals = _lazy("all_modals", lambda: get_all("category_2", "modal"))
all_auxs = _lazy("all_auxs", lambda: get_all("category_2", "aux"))
all_negated_modals_auxs = _lazy("all_negated_modals_auxs", lambda: get_all("negated", "1", all_modals_auxs))
all_non_negated_modals_auxs = _lazy("all_non_negated_modals_auxs", lambda: get_all("negated", "0", all_modals_auxs))
all_negated_modals = _lazy("all_negated_modals", lambda: get_all("negated", "1", all_modals))
all_non_negated_modals = _lazy("all_non_negated_modals", lambda: get_all("negated", "0", all_modals))
all_negated_auxs = _lazy("all_negated_auxs", lambda: get_all("negated", "1", all_auxs))
all_non_negated_auxs = _lazy("all_non_negated_auxs", lambda: get_all("negated", "0", all_auxs))

all_copulas = _lazy("all_copulas", lambda: get_all("category_2", "copula"))
all_finite_copulas = _lazy("all_finite_copulas", lambda: table_setdiff1d(all_copulas, get_all("bare", "1")))
all_rogatives = _lazy("all_rogatives", lambda: get_all("category", "(S\\NP)/Q"))

all_agreeing_aux = _lazy("all_agreeing_aux", lambda: table_setdiff1d(all_auxs, get_all("arg_1", "sg=1;sg=0")))
all_non_negative_agreeing_aux = _lazy("all_non_negative_agreeing_aux", lambda: get_all("negated", "0", all_agreeing_aux))
all_negative_agreeing_aux = _lazy("all_negative_agreeing_aux", lambda: get_all("negated", "1", all_agreeing_aux))
all_auxiliaries_no_null = _lazy("all_auxiliaries_no_null", lambda: table_setdiff1d(all_auxs, get_all("expression", "")))
all_non_negative_copulas = _lazy("all_non_negative_copulas", lambda: get_all("negated", "0", all_finite_copulas))
all_negative_copulas = _lazy("all_negative_copulas", lambda: get_all("negated", "1", all_finite_copulas))

# OTHER
all_determiners = _lazy("all_determiners", lambda: get_all("category", "(S/(S\\NP))/N"))
all_frequent_determiners = _lazy("all_frequent_determiners", lambda: get_all("frequent", "1", all_determiners))
all_very_common_dets = _lazy(
    "all_very_common_dets",
    lambda: np.append(get_all("expression", "the"), np.append(get_all("expression", "a"), get_all("expression", "an"))),
)
all_relativizers = _lazy("all_relativizers", lambda: get_all("category_2", "rel"))
all_reflexives = _lazy("all_reflexives", lambda: get_all("category_2", "refl"))
all_ACCpronouns = _lazy("all_ACCpronouns", lambda: get_all("category_2", "proACC"))
all_NOMpronouns = _lazy("all_NOMpronouns", lambda: get_all("category_2", "proNOM"))
all_embedding_verbs = _lazy("all_embedding_verbs", lambda: get_all("category_2", "V_embedding"))
all_wh_words = _lazy("all_wh_words", lambda: get_all("category", "NP_wh"))
all_demonstratives = _lazy(
    "all_demonstratives",
    lambda: np.append(
        get_all("expression", "this"),
        np.append(
            get_all_conjunctive([("category_2", "D"), ("expression", "that")]),
            np.append(get_all("expression", "these"), get_all("expression", "those")),
        ),
    ),
)
all_adjectives = _lazy(
    "all_adjectives",
    lambda: np.append(get_all("category_2", "adjective"), get_all("category_2", "Adj")),
)
all_frequent = _lazy("all_frequent", lambda: get_all("frequent", "1"))


__all__ = [name for name in globals() if name.startswith("all_")]
