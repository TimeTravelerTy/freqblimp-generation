from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional, Sequence

import numpy as np

from utils.randomize import get_active_policy
from utils.vocab_table import (
    _table_cache_key,
    get_all,
    get_table_zipf_expression,
    register_query_cache_clear_hook,
    table_setdiff1d,
    table_union1d,
)


_SUBJECT_RAISING_VERBS = (
    "seem", "appear", "happen", "tend", "look", "prove", "remain", "continue",
    "begin", "start", "stop", "fail", "threaten", "promise", "cease", "come",
    "grow", "commence", "chance", "transpire", "hap", "bid fair", "loom", "deserve",
    "turn out",
)

_OBJECT_RAISING_VERBS = (
    "want", "need", "like", "expect", "hate", "prefer", "mean", "wish", "imagine",
    "guess", "suppose", "reckon", "believe", "find", "feel", "know", "consider",
    "allow", "cause", "understand", "assume", "presume", "suspect", "claim",
    "maintain", "declare", "state", "report", "judge", "hold", "deem", "view",
    "perceive", "observe", "note", "reveal", "prove", "show", "require", "permit",
    "order", "command", "forbid", "authorize", "intend", "discover", "estimate",
    "calculate", "acknowledge", "admit", "affirm", "allege", "anticipate",
    "apprehend", "ascertain", "assert", "avow", "certify", "conceive", "conjecture",
    "deduce", "demonstrate", "determine", "discern", "enable", "envisage",
    "envision", "foresee", "grant", "guarantee", "hypothesize", "infer", "posit",
    "postulate", "predict", "proclaim", "pronounce", "recognize", "recollect",
    "remember", "stipulate", "surmise", "verify", "warrant",
)

_CONTROL_OBJECT_VERBS = (
    "advise", "ask", "beg", "command", "commission", "compel", "convince", "dare",
    "encourage", "entice", "force", "induce", "motivate", "obligate", "oblige",
    "order", "persuade", "press", "pressure", "prod", "prompt", "push", "require",
    "spur", "sway", "tempt", "urge",
)

_CONTROL_SUBJECT_VERBS = (
    "attempt", "bother", "continue", "fail", "hope", "intend", "like", "long",
    "manage", "need", "neglect", "plan", "propose", "try", "want", "wish", "yearn",
)

_RAISING_ADJECTIVES = (
    # core modal/dispositional
    "likely", "unlikely", "sure", "certain", "supposed", "going", "bound", "apt",
    "liable", "set", "due", "prone", "wont",
    # fate/disposition
    "fated", "destined", "doomed", "slated", "scheduled", "poised", "calculated",
    # canonical epistemic passives (unambiguous subject-raising reading)
    "rumored", "reputed", "alleged", "expected", "anticipated", "predicted",
    "estimated", "projected", "forecast", "reported", "believed", "thought",
    "assumed", "known", "understood", "claimed", "presumed", "suspected", "deemed",
    "said", "acknowledged", "recognized", "accepted", "conceded", "noted",
    "asserted", "established", "demonstrated",
    # excluded: deontic (required, needed, mandated, stipulated, specified),
    # attitudinal (feared, hoped), and marginal hypothetical/inferential
    # (hypothesized, posited, postulated, proposed, inferred, deduced)
)

_CONTROL_ADJECTIVES = (
    "eager", "ready", "willing", "unable", "able", "happy", "sad", "afraid",
    "anxious", "excited", "careful", "lucky", "reluctant", "hesitant", "keen",
    "loath", "quick", "slow", "prepared", "determined", "motivated", "impatient",
    "intent", "disinclined", "indisposed", "solicitous", "studious", "importunate",
    "powerless", "impotent", "desperate", "resolved", "fain", "minded", "chary",
)

_TOUGH_ADJECTIVES = (
    "easy", "hard", "fun", "nice", "good", "bad", "cool", "great", "safe", "boring",
    "weird", "tough", "simple", "scary", "sweet", "pleasant", "unpleasant", "lovely",
    "wonderful", "terrible", "horrible", "awful", "difficult", "dangerous",
    "impossible", "interesting", "exciting", "amazing", "relaxing", "stressful",
    "awkward", "strange", "tiring", "painful", "annoying", "tricky", "risky",
    "useful", "pointless", "delightful", "enjoyable", "fascinating", "engaging",
    "entertaining", "challenging", "complicated", "confusing", "frustrating",
    "exhausting", "draining", "demanding", "intimidating", "charming", "refreshing",
    "comforting", "soothing", "calming", "satisfying", "disappointing", "disturbing",
    "shocking", "embarrassing", "frightening", "tedious", "tiresome", "dreadful",
    "costly", "wasteful", "stimulating", "enlightening", "educational",
    "informative", "illuminating", "effortless", "breezy", "laborious", "arduous",
    "burdensome", "onerous", "strenuous", "exasperating", "infuriating", "maddening",
    "monotonous", "unbearable", "intolerable", "insufferable", "excruciating",
    "hazardous", "treacherous", "perilous", "formidable", "loathsome", "repulsive",
    "irksome", "wearisome", "cumbersome", "unsettling", "unnerving", "disarming",
    "invigorating", "uplifting", "edifying", "fruitless", "futile",
)

_EXISTENTIAL_CONTROL_SUBJECT_EXCLUDED_VERBS = (
    "continue", "fail", "need",
)

_DROP_ARGUMENT_GOOD_VERBS = (
    # original core list
    "aid", "approach", "ascend", "attack", "buy", "clean", "climb down",
    "climb up", "descend", "exit", "explore", "forget", "harm", "help",
    "hinder", "investigate", "know", "leave", "love", "observe", "pass",
    "purchase", "remember", "run around", "see", "sell", "skate around",
    "tour", "visit", "watch",
    # high-frequency absolute-use verbs added to base vocab (strict_trans → 0)
    "draw", "drink", "eat", "grow", "ride", "sing", "teach", "write",
    # additional verbs confirmed in overlay vocabulary with strict_trans=0
    # investigation / analysis
    "analyse", "analyze", "annotate", "assess", "audit", "calibrate",
    "classify", "configure", "debug", "diagnose", "document", "evaluate",
    "inspect", "monitor", "proofread", "register", "scrutinize", "sort",
    "track", "transcribe",
    # cleaning / maintenance
    "buff", "disinfect", "insulate", "lubricate", "polish", "scrub",
    "shellac", "sterilize", "trim",
    # agriculture / horticulture
    "cultivate", "plant", "reap", "weed",
    # crafts / trade / provisioning
    "garnish", "hoard", "marinate", "peddle", "pluck", "procure",
    "salvage", "scavenge", "season", "stockpile",
    # administration / services
    "advertise", "allocate", "counsel", "educate", "excavate", "import",
    "instruct", "post", "prepare", "quarry", "supervise", "treat",
)

_DROP_ARGUMENT_BAD_VERBS = (
    # conservative null-object-resistant transitive verbs
    "abandon", "address", "admire", "adopt", "affect",
    "alter", "apprehend", "arrest", "avoid", "banish", "betray", "blame",
    "block", "capture", "confiscate", "contain",
    "contradict", "convince", "criticize", "damage", "defeat", "delete",
    "demolish", "deny", "depict", "deprive", "destroy", "devour", "dismiss",
    "disturb", "elect", "embrace", "encounter", "endorse", "erase", "evict",
    "examine", "exclude", "greet", "guard", "humiliate", "identify",
    "ignore", "imitate", "imprison", "interrogate", "interrupt", "intimidate",
    "involve", "kidnap", "mention", "mock", "murder", "name", "overthrow",
    "persuade", "possess", "praise", "prevent", "punish", "question", "quote",
    "reject", "release", "replace", "rescue", "require", "ridicule", "rob",
    "seize", "select", "solve", "sue", "surprise", "survey",
    "transport", "undermine", "upset", "value", "verify", "witness",
)

_CAUSATIVE_ALTERNATING_VERBS = (
    # thermal / phase change
    "acidify", "bake", "burn", "caramelize", "char", "coagulate", "condense",
    "congeal", "curdle", "evaporate", "freeze", "liquefy", "melt", "overcook",
    "parch", "scald", "scorch", "soak", "solidify", "vaporize", "wash",
    # mechanical / structural change
    "calcify", "cheapen", "chip", "crack", "crumple", "decouple", "discolor",
    "dislocate", "distend", "entangle", "fade", "flatten", "fold", "fray",
    "granulate", "homogenize", "loosen", "roughen", "shatter", "shrink",
    "slacken", "smarten", "steepen", "stretch", "tighten", "twist", "unbuckle",
    "unbutton", "uncoil", "unfasten", "unfold", "unhook", "unseal", "warp",
    "wrinkle",
    # color / luminance change
    "blacken", "brighten", "darken", "dim",
    # motion / position change
    "close", "drop", "fling open", "hide", "hide away", "move", "open", "roll",
    "shut", "sit down", "slow", "speed up", "spin around", "stand up", "stop",
    "tip over", "turn", "wake up",
    # social / process change
    "accelerate", "assemble", "awaken", "change", "crash", "dry", "grow",
    "maneuver", "marry", "reunite", "steer", "train",
    # other
    "benefit", "worry",
)

COUNT_TRIGGERS = {
    "many", "few", "several", "both", "these", "those", "this", "that", "every",
    "each", "a", "an", "one", "two", "three", "four", "five",
}
MASS_TRIGGERS = {"much", "little", "less"}

_NUMBER_WORDS = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}
_STRANDED_PREP_SUFFIXES = (
    " with",
    " for",
    " to",
    " from",
    " at",
    " on",
    " in",
    " about",
    " of",
)


@dataclass(frozen=True)
class OverlayGuard:
    uid: str
    varying_slots: Sequence[str]
    shared_slots: Sequence[str]
    good_pool: str
    bad_pool: str


GUARD_REGISTRY = {
    "passive_1": OverlayGuard("passive_1", ("verb",), ("patient", "agent", "aux"), "passivizable_en", "nonpassivizable_en"),
    "passive_2": OverlayGuard("passive_2", ("verb",), ("patient", "aux"), "passivizable_en", "nonpassivizable_en"),
    "causative": OverlayGuard("causative", ("verb",), ("subject", "object", "aux"), "alternating", "strict_intransitive"),
    "inchoative": OverlayGuard("inchoative", ("verb",), ("subject", "aux"), "alternating", "strict_transitive"),
    "drop_argument": OverlayGuard("drop_argument", ("verb",), ("subject", "aux"), "drop_permitting", "drop_forbidden"),
    "animate_subject_passive": OverlayGuard("animate_subject_passive", ("agent_noun",), ("verb", "patient", "copula"), "animate_common_noun", "inanimate_common_noun"),
    "animate_subject_trans": OverlayGuard("animate_subject_trans", ("subject_noun",), ("verb", "object"), "animate_subject", "inanimate_subject"),
    "existential_there_subject_raising": OverlayGuard("existential_there_subject_raising", ("predicate",), ("there", "to_be", "determiner", "embedded_subject", "embedded_vp"), "raising_subject", "control_subject"),
    "existential_there_object_raising": OverlayGuard("existential_there_object_raising", ("matrix_verb",), ("matrix_subject", "aux", "there_to_be", "embedded_subject", "embedded_vp"), "raising_object", "control_object"),
    "expletive_it_object_raising": OverlayGuard("expletive_it_object_raising", ("matrix_verb",), ("matrix_subject", "aux", "it_to_be", "adjective", "that_clause"), "raising_object", "control_object"),
    "tough_vs_raising_1": OverlayGuard("tough_vs_raising_1", ("adjective",), ("subject", "copula", "infinitive"), "tough_adjective", "raising_adjective"),
    "tough_vs_raising_2": OverlayGuard("tough_vs_raising_2", ("adjective",), ("subject", "copula", "infinitive"), "raising_adjective", "tough_adjective"),
    "existential_there_quantifiers_1": OverlayGuard("existential_there_quantifiers_1", ("determiner",), ("there", "aux", "noun", "vp"), "existential_good_quantifier", "existential_bad_quantifier"),
    "existential_there_quantifiers_2": OverlayGuard("existential_there_quantifiers_2", ("word_order",), ("determiner", "noun", "aux", "vp"), "canonical_order", "existential_order"),
    "superlative_quantifiers_1": OverlayGuard("superlative_quantifiers_1", ("quantifier_phrase",), ("subject", "verb", "object"), "negative_superlative_good", "negative_superlative_bad"),
    "superlative_quantifiers_2": OverlayGuard("superlative_quantifiers_2", ("determiner",), ("subject_noun", "verb", "quantifier_phrase", "object"), "non_negative_determiner", "negative_determiner"),
    "ellipsis_n_bar_1": OverlayGuard("ellipsis_n_bar_1", ("ellipsis_site",), ("verb", "subjects", "object_noun", "determiners", "adjective"), "licensed_nbar_ellipsis", "misplaced_adjective"),
    "ellipsis_n_bar_2": OverlayGuard("ellipsis_n_bar_2", ("ellipsis_site",), ("verb", "subjects", "object_noun", "determiners", "adjectives"), "licensed_nbar_ellipsis", "misplaced_nbar_ellipsis"),
}


@lru_cache(maxsize=1)
def build_agreement_safe_verbs():
    """Union of pres/ing/en verbs, minus forms homophonous with a past tense.

    Replaces the O(N²) nested Python filter used in the agreement generators,
    which hangs when run with the large vocabulary overlay.
    """
    from functools import reduce as _reduce
    from utils.vocab_sets import all_verbs as _all_verbs

    safe_verbs = _reduce(np.union1d, (
        get_all("pres", "1", _all_verbs),
        get_all("ing", "1", _all_verbs),
        get_all("en", "1", _all_verbs),
    ))
    past_mask = _all_verbs["past"] == "1"
    past_root_expr = set(zip(
        np.asarray(_all_verbs["root"][past_mask], dtype=str),
        np.asarray(_all_verbs["expression"][past_mask], dtype=str),
    ))
    pres_verbs = get_all("pres", "1", _all_verbs)
    is_ambiguous = np.array(
        [(str(v["root"]), str(v["expression"])) in past_root_expr for v in pres_verbs],
        dtype=bool,
    )
    if np.any(is_ambiguous):
        safe_verbs = np.setdiff1d(safe_verbs, pres_verbs[is_ambiguous])
    return safe_verbs


@lru_cache(maxsize=1)
def non_past_verb_rows():
    from utils.vocab_sets import all_verbs as _all_verbs

    return table_setdiff1d(_all_verbs, get_all("past", "1", _all_verbs))


def overlay_enabled() -> bool:
    policy = get_active_policy()
    return bool(policy and getattr(policy, "overlay_enabled", False))


_ZIPF_FILTER_CACHE: dict = {}


def _clear_zipf_filter_cache():
    _ZIPF_FILTER_CACHE.clear()


register_query_cache_clear_hook(_clear_zipf_filter_cache)


def filter_rows_for_active_zipf(table, controlled_pos: str, fallback_on_empty: bool=True):
    policy = get_active_policy()
    if policy is None or len(table) == 0:
        return table
    if controlled_pos not in getattr(policy, "controlled_pos_set", set()):
        return table
    lower, upper = policy.bounds_for(controlled_pos)
    if lower is None and upper is None:
        return table
    cache_key = (_table_cache_key(table), controlled_pos, lower, upper, bool(fallback_on_empty))
    cached = _ZIPF_FILTER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    zipf_values = get_table_zipf_expression(table)
    mask = np.ones(len(table), dtype=bool)
    if lower is not None:
        mask &= zipf_values >= lower
    if upper is not None:
        mask &= zipf_values <= upper
    filtered = table[mask]
    if len(filtered) == 0 and fallback_on_empty:
        filtered = table
    _ZIPF_FILTER_CACHE[cache_key] = filtered
    return filtered


_INFLECTION_FIELDS = ("finite", "bare", "pres", "past", "ing", "en", "3sg")


def rows_matching_inflection(table, template_row):
    if len(table) == 0:
        return table
    mask = np.ones(len(table), dtype=bool)
    for field in _INFLECTION_FIELDS:
        mask &= np.asarray(table[field], dtype=str) == str(template_row[field])
    return table[mask]


def rows_matching_expressions(table, allowed_rows):
    if len(table) == 0:
        return table
    if len(allowed_rows) == 0:
        return table[:0]
    allowed = np.unique(np.asarray(allowed_rows["expression"], dtype=str))
    return table[np.isin(np.asarray(table["expression"], dtype=str), allowed)]


@lru_cache(maxsize=4096)
def _expression_surface_variants(expression: str):
    text = str(expression).strip().lower()
    if not text:
        return ()
    pieces = text.split(" ", 1)
    head = pieces[0]
    suffix = pieces[1] if len(pieces) == 2 else ""
    variants = {text}
    try:
        from lemminflect import getInflection
    except Exception:
        return tuple(sorted(variants))
    for tag in ("VB", "VBP", "VBZ", "VBG", "VBD", "VBN"):
        for form in getInflection(head, tag=tag) or ():
            form = str(form).strip().lower()
            if not form:
                continue
            variants.add("%s %s" % (form, suffix) if suffix else form)
    return tuple(sorted(variants))


def _rows_for_expression_families(table, expressions: Sequence[str], expand_inflections: bool=False):
    normalized_expressions = set()
    for expression in expressions:
        variants = _expression_surface_variants(expression) if expand_inflections else (expression,)
        normalized_expressions.update(
            str(variant).strip()
            for variant in variants
            if str(variant).strip()
        )
    if len(table) == 0 or not normalized_expressions:
        return np.array([], dtype=getattr(table, "dtype", None))

    expression_values = np.asarray(table["expression"], dtype=str)
    exact_mask = np.isin(expression_values, sorted(normalized_expressions))
    if not np.any(exact_mask):
        return np.array([], dtype=getattr(table, "dtype", None))
    return table[exact_mask]


def _exclude_expression_families(table, expressions: Sequence[str]):
    if len(table) == 0:
        return table
    normalized_expressions = {str(expression).strip() for expression in expressions if str(expression).strip()}
    excluded_roots = set()
    for expression in normalized_expressions:
        exact = get_all("expression", expression, table)
        for row in exact:
            root = str(row["root"]).strip()
            if root:
                excluded_roots.add(root)
    keep_mask = np.ones(len(table), dtype=bool)
    if normalized_expressions:
        keep_mask &= ~np.isin(np.asarray(table["expression"], dtype=str), list(normalized_expressions))
    if excluded_roots:
        keep_mask &= ~np.isin(np.asarray(table["root"], dtype=str), list(excluded_roots))
    filtered = table[keep_mask]
    return filtered if len(filtered) > 0 else table


def raising_verb_list_for_uid(uid: str) -> Sequence[str]:
    uid = str(uid or "").strip().lower()
    if "object_raising" in uid:
        return _OBJECT_RAISING_VERBS
    return _SUBJECT_RAISING_VERBS


def guard_for_uid(uid: str) -> Optional[OverlayGuard]:
    return GUARD_REGISTRY.get(str(uid or "").strip().lower())


def verb_rows_for_category(category_2: str):
    return get_all("category_2", category_2)


def adjective_rows_for_category(category_2: str):
    return get_all("category_2", category_2)


def subject_raising_verb_rows():
    rows = verb_rows_for_category("V_raising_subj")
    return _rows_for_expression_families(rows, _SUBJECT_RAISING_VERBS, expand_inflections=True)


def control_subject_verb_rows():
    rows = verb_rows_for_category("V_control_subj")
    return _rows_for_expression_families(rows, _CONTROL_SUBJECT_VERBS, expand_inflections=True)


def object_raising_verb_rows():
    rows = verb_rows_for_category("V_raising_object")
    return _rows_for_expression_families(rows, _OBJECT_RAISING_VERBS, expand_inflections=True)


def control_object_verb_rows():
    rows = verb_rows_for_category("V_control_object")
    return _rows_for_expression_families(rows, _CONTROL_OBJECT_VERBS, expand_inflections=True)


def subject_raising_adjective_rows():
    """Return only the canonical subject-raising adjectives.

    Filters vocabulary rows by _RAISING_ADJECTIVES so that the curated list
    is the single source of truth for both generators and the overlay pipeline.
    """
    rows = adjective_rows_for_category("Adj_raising_subj")
    return _rows_for_expression_families(rows, _RAISING_ADJECTIVES)


def control_subject_adjective_rows():
    rows = adjective_rows_for_category("Adj_control_subj")
    return _rows_for_expression_families(rows, _CONTROL_ADJECTIVES)


def tough_adjective_rows():
    return adjective_rows_for_category("Adj_tough")


def clausal_it_adjective_rows():
    rows = adjective_rows_for_category("Adj_clausal")
    return rows[rows["arg_1"] == "expression=it"]


def existential_bad_control_subject_verb_rows():
    rows = control_subject_verb_rows()
    return _exclude_expression_families(rows, _EXISTENTIAL_CONTROL_SUBJECT_EXCLUDED_VERBS)


def drop_argument_good_verb_rows():
    rows = _rows_for_expression_families(get_all("verb", "1"), _DROP_ARGUMENT_GOOD_VERBS, expand_inflections=True)
    return get_all("strict_trans", "0", rows)


def drop_argument_bad_verb_rows():
    return _rows_for_expression_families(
        get_all("category", "(S\\NP)/NP"),
        _DROP_ARGUMENT_BAD_VERBS,
        expand_inflections=True,
    )


def causative_alternating_verb_rows():
    """Return only genuine causative-alternating verb rows (TV and IV forms).

    Filters both causative=1 and inchoative=1 rows through _CAUSATIVE_ALTERNATING_VERBS,
    excluding overlay verbs that incorrectly inherit causative/inchoative flags from
    unrelated base verbs (e.g. protrude inheriting from accelerate).
    """
    rows = _rows_for_expression_families(
        get_all("verb", "1"),
        _CAUSATIVE_ALTERNATING_VERBS,
        expand_inflections=True,
    )
    filtered = table_union1d(get_all("causative", "1", rows), get_all("inchoative", "1", rows))
    return filtered if len(filtered) > 0 else rows


def requirement_from_text(*parts: Iterable[object]) -> Optional[str]:
    toks = set()
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            for item in part:
                if item is not None:
                    toks.update(str(item).lower().split())
        else:
            toks.update(str(part).lower().split())
    if toks & COUNT_TRIGGERS:
        return "COUNT"
    if toks & MASS_TRIGGERS:
        return "MASS"
    return None


def filter_nouns_for_requirement(table, req: Optional[str]):
    if req == "COUNT":
        return get_all("mass", "0", table)
    if req == "MASS":
        return get_all("mass", "1", table)
    return table


def filter_stranded_preposition_verbs(table):
    if len(table) == 0:
        return table
    expressions = np.asarray(table["expression"], dtype=str)
    keep_mask = np.ones(len(expressions), dtype=bool)
    for suffix in _STRANDED_PREP_SUFFIXES:
        keep_mask &= ~np.char.endswith(expressions, suffix)
    filtered = table[keep_mask]
    return filtered if len(filtered) > 0 else table


@lru_cache(maxsize=4096)
def noun_expression_looks_plural(expression: str) -> bool:
    text = str(expression or "").strip().lower()
    if not text or " " in text:
        return False
    try:
        from lemminflect import getLemma
    except Exception:
        return False
    lemmas = tuple(str(lemma).strip().lower() for lemma in (getLemma(text, upos="NOUN") or ()))
    if not lemmas:
        return False
    return all(lemma and lemma != text for lemma in lemmas)


def filter_plural_looking_singular_nouns(table):
    if len(table) == 0:
        return table
    expr = np.asarray(table["expression"], dtype=str)
    sg = np.asarray(table["sg"], dtype=str)
    suspicious = (sg == "1") & np.fromiter(
        (noun_expression_looks_plural(expression) for expression in expr),
        dtype=bool,
        count=len(expr),
    )
    filtered = table[~suspicious]
    return filtered if len(filtered) > 0 else table


def number_to_words(value: int) -> str:
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError("number_to_words expects an integer-like value")
    if value not in _NUMBER_WORDS:
        raise ValueError("Unsupported number for local number_to_words: %s" % value)
    return _NUMBER_WORDS[value]


def curated_template_expressions(category_2: str) -> Sequence[str]:
    if category_2 == "Adj_raising_subj":
        return _RAISING_ADJECTIVES
    if category_2 == "Adj_control_subj":
        return _CONTROL_ADJECTIVES
    if category_2 == "Adj_tough":
        return _TOUGH_ADJECTIVES
    if category_2 == "V_raising_subj":
        return _SUBJECT_RAISING_VERBS
    if category_2 == "V_raising_object":
        return _OBJECT_RAISING_VERBS
    return ()
