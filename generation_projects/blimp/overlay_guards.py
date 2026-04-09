from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from utils.randomize import get_active_policy
from utils.vocab_table import get_all, get_table_zipf_expression


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

_RAISING_ADJECTIVES = (
    "likely", "unlikely", "sure", "certain", "supposed", "going", "bound", "apt",
    "liable", "set", "due", "prone", "fated", "destined", "doomed", "slated",
    "scheduled", "wont", "calculated", "rumored", "reputed", "alleged", "poised",
    "expected", "anticipated", "predicted", "estimated", "projected", "forecast",
    "reported", "believed", "thought", "assumed", "known", "understood", "claimed",
    "presumed", "suspected", "deemed", "said", "acknowledged", "recognized",
    "accepted", "conceded", "noted", "asserted", "inferred", "deduced",
    "established", "demonstrated", "posited", "postulated", "hypothesized",
    "proposed", "required", "needed", "mandated", "stipulated", "specified",
    "feared", "hoped", "soon", "about",
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
    "drop_argument": OverlayGuard("drop_argument", ("verb",), ("subject", "aux"), "drop_permitting", "strict_transitive"),
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


def overlay_enabled() -> bool:
    policy = get_active_policy()
    return bool(policy and getattr(policy, "overlay_enabled", False))


def filter_rows_for_active_zipf(table, controlled_pos: str, fallback_on_empty: bool=True):
    policy = get_active_policy()
    if policy is None or len(table) == 0:
        return table
    if controlled_pos not in getattr(policy, "controlled_pos_set", set()):
        return table
    lower, upper = policy.bounds_for(controlled_pos)
    if lower is None and upper is None:
        return table
    zipf_values = get_table_zipf_expression(table)
    mask = np.ones(len(table), dtype=bool)
    if lower is not None:
        mask &= zipf_values >= lower
    if upper is not None:
        mask &= zipf_values <= upper
    filtered = table[mask]
    if len(filtered) > 0 or not fallback_on_empty:
        return filtered
    return table


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
    return verb_rows_for_category("V_raising_subj")


def control_subject_verb_rows():
    return verb_rows_for_category("V_control_subj")


def object_raising_verb_rows():
    return verb_rows_for_category("V_raising_object")


def control_object_verb_rows():
    return verb_rows_for_category("V_control_object")


def subject_raising_adjective_rows():
    return adjective_rows_for_category("Adj_raising_subj")


def control_subject_adjective_rows():
    return adjective_rows_for_category("Adj_control_subj")


def tough_adjective_rows():
    return adjective_rows_for_category("Adj_tough")


def clausal_it_adjective_rows():
    rows = adjective_rows_for_category("Adj_clausal")
    return rows[rows["arg_1"] == "expression=it"]


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
