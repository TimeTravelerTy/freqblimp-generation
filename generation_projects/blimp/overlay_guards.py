import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import wn

from utils.vocab_sets import all_nominals
from utils.exceptions import LexicalGapError
from utils.randomize import get_active_policy, uniform_choice
from utils.vocab_table import (
    _table_cache_key,
    get_all,
    get_all_conjunctive,
    get_matched_by,
    get_matches_of,
    get_table_zipf_expression,
    register_query_cache_clear_hook,
    table_setdiff1d,
    table_union1d,
)


_SUBJECT_RAISING_VERBS = (
    "seem", "appear", "happen", "tend", "look", "prove", "remain", "continue",
    "begin", "start", "stop", "fail", "threaten", "promise", "cease", "come",
    "grow", "commence", "chance", "deserve", "turn out",
)

_OBJECT_RAISING_VERBS = (
    "anticipate", "believe", "consider", "declare", "determine", "discover",
    "expect", "find", "judge", "need", "predict", "prefer", "want",
)

_CONTROL_OBJECT_VERBS = (
    "advise", "ask", "beg", "commission", "compel", "convince", "dare",
    "encourage", "entice", "induce", "motivate", "obligate", "oblige",
    "persuade", "press", "pressure", "prod", "prompt", "push",
    "spur", "sway", "tempt", "urge",
)

_PASSIVE_BAD_VERB_BLOCKLIST = (
    # Otherwise intransitive inventory entries whose participles can surface as
    # acceptable adjectival or transitive passives in the passive templates.
    "appeal", "decease", "expire", "piss", "speak",
)

_PASSIVE_GOOD_VERB_BLOCKLIST = (
    # Lemminflect can admit bare-looking participles here ("was bust").
    "bust",
)

_INTRANSITIVE_CONTRAST_BLOCKLIST = _PASSIVE_BAD_VERB_BLOCKLIST + (
    # Direct-object uses are common enough to blur transitive/passive contrasts.
    "side",
    # These strongly prefer PP complements in this template; bare clauses like
    # "the poet resulted" or "the guests resort" are not good simple
    # intransitivity minimal pairs.
    "resort", "result",
)

_TRANSITIVE_CONTRAST_AMBIGUOUS_BLOCKLIST = (
    # These have ordinary objectless uses that make them poor bad-side choices
    # for intransitive minimal pairs, even if the inventory currently marks
    # only a transitive frame.
    "blaspheme", "bluff", "interpose", "shoplift",
)

_NONPASSIVIZABLE_PARTICIPLE_VERBS = (
    # Original-BLiMP-style intransitives whose participles stay bad in passive
    # templates, used to avoid the head regime collapsing to a tiny repeated set.
    "arrive", "boast", "chat", "compete", "complain", "cough", "cry",
    "dance", "die", "disagree", "emerge", "exist", "flirt", "happen",
    "laugh", "lie", "occur", "proceed", "progress", "react", "reply",
    "respond", "scream", "sleep", "smile", "sneeze", "struggle",
    "testify", "vanish", "wait", "yawn",
)

_EXISTENTIAL_SUBJECT_RAISING_VERBS = (
    # Verbal predicates that are natural in the "there ... to be" existential
    # frame. Broader subject-raising verbs like "promise" or "remain" are kept
    # available for other raising paradigms but read poorly here.
    "appear", "come", "continue", "fail", "happen", "look", "prove", "seem",
    "tend", "turn out",
)

_EXISTENTIAL_SUBJECT_RAISING_ADJECTIVES = (
    # Conservative high-confidence raising predicates for existential there.
    # Excludes marginal/passive-reporting items such as "conceded" and
    # "accepted", which are acceptable in other templates but not here.
    "believed", "bound", "certain", "due", "estimated", "expected",
    "forecast", "known", "likely", "projected", "reported", "said",
    "scheduled", "supposed", "thought", "unlikely",
)

_LOW_QUALITY_OVERLAY_VERB_LEMMAS = (
    # Mostly noun/adjective-derived or name-like verb uses whose wordfreq score
    # is driven by non-verbal senses. They are valid dictionary verbs, but they
    # make simple BLiMP argument-structure templates read as lexical oddities.
    "antique", "bucket", "charcoal", "harry", "hay", "holiday",
    "honeymoon", "hull", "low", "nut", "pearl", "season", "summer",
    "taxi", "tool", "twitter", "vacation", "war", "weekend", "winter",
)

_LOW_QUALITY_NOMINAL_EXPRESSIONS = (
    # Nominalized adjectives that surface poorly with arbitrary determiners.
    "airs", "altogether", "colonial", "contemporary", "dependent",
    "dining", "disabled", "divine", "drinkable", "empty", "federal",
    "homosexual", "immune", "independent", "invalid", "moderate", "poor",
    "prior", "probable", "posing", "rich", "romantic", "semitic", "silly", "skating",
    "temporary",
)

_AGENTIVE_VERB_SUBJECT_DOMAINS = frozenset({
    "verb.body",
    "verb.cognition",
    "verb.communication",
    "verb.competition",
    "verb.consumption",
    "verb.emotion",
    "verb.possession",
    "verb.social",
})

_AGENTIVE_ARG_MARKERS = (
    "animal=1",
    "animate=1",
    "institution=1",
    "person=1",
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
    "loath", "quick", "slow", "prepared", "motivated", "impatient",
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

_CLAUSAL_IT_ADJECTIVES = (
    "a shame", "acceptable", "amusing", "apparent", "bad", "boring", "clear",
    "cool", "expected", "fortunate", "funny", "good", "important",
    "interesting", "lucky", "natural", "nice", "normal", "not so amusing",
    "not so bad", "not so clear", "not so funny", "not so good",
    "not so important", "not so interesting", "not so nice", "not so normal",
    "not so obvious", "not so pleasant", "not so surprising", "not so weird",
    "noteworthy", "obvious", "odd", "okay", "pleasant", "sad", "strange",
    "surprising", "too bad", "unexpected", "unfortunate", "uninteresting",
    "unlucky", "unpleasant", "unsurprising", "unusual", "weird",
    "worth mentioning", "worth noting", "worth saying",
)

_FINITE_CLAUSE_EMBEDDING_VERBS = (
    "admit", "argue", "assert", "believe", "claim", "confess", "conclude",
    "deny", "discover", "explain", "feel", "forget", "hope", "imagine",
    "know", "learn", "notice", "remember", "reveal", "say", "think",
    "whisper",
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

_CAUSATIVE_BAD_EXTRA_INTRANSITIVES = (
    "appear", "disappear", "emerge", "exist", "happen", "occur", "remain",
    "vanish",
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


def mismatching_nonpast_agreement_form(verb_row):
    """Return the opposite 3sg present-tense form for the same root.

    The lexical item has already been chosen when `verb_row` is sampled, so the
    disagreement form should not trigger another Zipf-constrained lexical draw.
    Using uniform choice here removes a major retry hotspot in subject-verb
    agreement generators under narrow regimes like xtail.
    """
    if verb_row["finite"] != "1":
        return verb_row
    target_3sg = "0" if verb_row["3sg"] == "1" else "1"
    alt_forms = get_all_conjunctive(
        [("pres", "1"), ("3sg", target_3sg)],
        get_all("root", verb_row["root"]),
    )
    if len(alt_forms) == 0:
        raise LexicalGapError(
            "No mismatching non-past agreement form for root=%s" % verb_row["root"]
        )
    return uniform_choice(alt_forms)


def overlay_enabled() -> bool:
    policy = get_active_policy()
    return bool(policy and getattr(policy, "overlay_enabled", False))


_ZIPF_FILTER_CACHE: dict = {}
_CURATED_ZIPF_MIN_CANDIDATES = 10
_VERB_INVENTORY_PATH = Path(__file__).with_name("verb_inventory.json")


def _clear_zipf_filter_cache():
    _ZIPF_FILTER_CACHE.clear()
    _transitive_source_lemmas.cache_clear()
    _inventory_core_frame_lemmas.cache_clear()
    _inventory_any_intransitive_lemmas.cache_clear()
    _inventory_core_intr_lemmas.cache_clear()
    _inventory_core_trans_lemmas.cache_clear()
    _verb_subject_domains.cache_clear()


register_query_cache_clear_hook(_clear_zipf_filter_cache)


def _source_lemma_for_row(row) -> str:
    root = str(row["root"]).strip() if "root" in row.dtype.names else ""
    if "_overlay_" in root:
        root = root.split("_overlay_", 1)[0]
    elif "_" in root:
        stem, suffix = root.rsplit("_", 1)
        if "\\" in suffix or "/" in suffix or suffix in {"S", "NP", "N"}:
            root = stem
    return (root or str(row["expression"]).strip()).replace("_", " ")


@lru_cache(maxsize=1)
def _transitive_source_lemmas():
    transitive_rows = get_all("category", "(S\\NP)/NP", get_all("verb", "1"))
    return frozenset(_source_lemma_for_row(row) for row in transitive_rows)


def exclude_transitive_source_lemmas(table):
    """Keep only rows whose source lemma has no transitive frame in the active vocab.

    Overlay verb rows inherit many BLiMP template signatures per coarse frame.
    A lemma such as "dominate" can therefore have both a transitive/passive
    template and a strict-intransitive template. Row-level strict_intrans=1 is
    not enough for paradigms that require a genuinely non-transitive contrast.
    """
    if len(table) == 0:
        return table
    transitive_lemmas = _transitive_source_lemmas()
    keep = np.fromiter(
        (_source_lemma_for_row(row) not in transitive_lemmas for row in table),
        dtype=bool,
        count=len(table),
    )
    return table[keep]


@lru_cache(maxsize=1)
def _inventory_core_frame_lemmas():
    if not _VERB_INVENTORY_PATH.exists():
        return frozenset(), frozenset()
    with _VERB_INVENTORY_PATH.open() as handle:
        payload = json.load(handle)
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    core_intr = set()
    core_trans = set()
    for entry in entries:
        lemma = str(entry.get("lemma", "")).strip().lower()
        if not lemma:
            continue
        for frame in entry.get("frames", ()):
            kind = frame.get("type") or frame.get("kind")
            if kind == "intr":
                core_intr.add(lemma)
            elif kind == "trans" or (isinstance(kind, str) and kind.startswith("trans")):
                core_trans.add(lemma)
    return frozenset(core_intr), frozenset(core_trans)


@lru_cache(maxsize=1)
def _inventory_core_intr_lemmas():
    return _inventory_core_frame_lemmas()[0]


@lru_cache(maxsize=1)
def _inventory_core_trans_lemmas():
    return _inventory_core_frame_lemmas()[1]


@lru_cache(maxsize=None)
def _verb_subject_domains(lemma):
    try:
        synsets = wn.synsets(lemma, pos="v", lexicon="oewn:2021")
    except Exception:
        return frozenset()
    domains = set()
    for synset in synsets:
        metadata = synset.metadata() if callable(getattr(synset, "metadata", None)) else {}
        subject = metadata.get("subject") if isinstance(metadata, dict) else None
        if subject:
            domains.add(subject)
    return frozenset(domains)


def _verb_requires_agentive_subject(lemma):
    domains = _verb_subject_domains(lemma)
    return bool(domains) and domains.issubset(_AGENTIVE_VERB_SUBJECT_DOMAINS)


def _row_allows_agentive_subject(row):
    requirement = str(row["arg_1"]) if "arg_1" in row.dtype.names else ""
    return any(marker in requirement for marker in _AGENTIVE_ARG_MARKERS)


def exclude_agentive_subject_mismatch_rows(table):
    if len(table) == 0:
        return table
    keep = np.fromiter(
        (
            not _verb_requires_agentive_subject(_source_lemma_for_row(row).lower())
            or _row_allows_agentive_subject(row)
            for row in table
        ),
        dtype=bool,
        count=len(table),
    )
    return table[keep]


@lru_cache(maxsize=1)
def _inventory_any_intransitive_lemmas():
    if not _VERB_INVENTORY_PATH.exists():
        return frozenset()
    with _VERB_INVENTORY_PATH.open() as handle:
        payload = json.load(handle)
    entries = payload.get("entries", payload) if isinstance(payload, dict) else payload
    lemmas = set()
    for entry in entries:
        lemma = str(entry.get("lemma", "")).strip().lower()
        if not lemma:
            continue
        for frame in entry.get("frames", ()):
            kind = frame.get("type") or frame.get("kind")
            if isinstance(kind, str) and (kind == "intr" or kind.startswith("intr_")):
                lemmas.add(lemma)
                break
    return frozenset(lemmas)


def exclude_source_lemmas(table, lemmas):
    if len(table) == 0:
        return table
    lemma_set = set(lemmas)
    if not lemma_set:
        return table
    keep = np.fromiter(
        (_source_lemma_for_row(row).lower() not in lemma_set for row in table),
        dtype=bool,
        count=len(table),
    )
    return table[keep]


def exclude_low_quality_overlay_verb_lemmas(table):
    return exclude_source_lemmas(table, _LOW_QUALITY_OVERLAY_VERB_LEMMAS)


def safe_transitive_verb_rows():
    rows = get_all("category", "(S\\NP)/NP", get_all("verb", "1"))
    return exclude_low_quality_overlay_verb_lemmas(rows)


def safe_intransitive_verb_rows():
    rows = get_all("category", "S\\NP", get_all("verb", "1"))
    rows = exclude_agentive_subject_mismatch_rows(rows)
    return exclude_low_quality_overlay_verb_lemmas(rows)


def strict_zipf_rows(table, controlled_pos: str, minimum_candidates: Optional[int]=None):
    return filter_rows_for_active_zipf(
        table,
        controlled_pos,
        fallback_on_empty=False,
        minimum_candidates=minimum_candidates,
    )


def _zipf_distance_from_window(zipf_values, lower, upper):
    distance = np.zeros(len(zipf_values), dtype=float)
    if lower is not None:
        distance = np.maximum(distance, lower - zipf_values)
    if upper is not None:
        distance = np.maximum(distance, zipf_values - upper)
    return distance


def filter_rows_for_active_zipf(table, controlled_pos: str, fallback_on_empty: bool=True, minimum_candidates: Optional[int]=None):
    policy = get_active_policy()
    if policy is None or len(table) == 0:
        return table
    if controlled_pos not in getattr(policy, "controlled_pos_set", set()):
        return table
    lower, upper = policy.bounds_for(controlled_pos)
    if lower is None and upper is None:
        return table
    target_count = None
    if minimum_candidates is not None:
        target_count = max(int(minimum_candidates), 0)
    cache_key = (
        _table_cache_key(table),
        controlled_pos,
        lower,
        upper,
        bool(fallback_on_empty),
        target_count,
    )
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
    if target_count:
        target_count = min(target_count, len(table))
        if len(filtered) < target_count:
            distance = _zipf_distance_from_window(zipf_values, lower, upper)
            order = np.argsort(distance, kind="stable")
            threshold = float(distance[order[target_count - 1]])
            filtered = table[distance <= threshold]
    if len(filtered) == 0 and fallback_on_empty:
        filtered = table
    _ZIPF_FILTER_CACHE[cache_key] = filtered
    return filtered


def choose_row_for_active_zipf(table,
                               controlled_pos: str,
                               fallback_on_empty: bool=False,
                               avoid=None,
                               error_message: Optional[str]=None,
                               minimum_candidates: Optional[int]=None):
    candidates = filter_rows_for_active_zipf(
        table,
        controlled_pos,
        fallback_on_empty=fallback_on_empty,
        minimum_candidates=minimum_candidates,
    )
    if len(candidates) == 0:
        raise LexicalGapError(error_message or "No %s candidates available" % controlled_pos)
    return uniform_choice(candidates, avoid=avoid)


def choose_row_for_active_zipf_by_source_lemma(table,
                                               controlled_pos: str,
                                               fallback_on_empty: bool=False,
                                               avoid=None,
                                               error_message: Optional[str]=None,
                                               minimum_candidates: Optional[int]=None):
    candidates = filter_rows_for_active_zipf(
        table,
        controlled_pos,
        fallback_on_empty=fallback_on_empty,
        minimum_candidates=minimum_candidates,
    )
    if len(candidates) == 0:
        raise LexicalGapError(error_message or "No %s candidates available" % controlled_pos)
    if avoid is not None:
        try:
            avoid_exprs = set(np.atleast_1d(np.asarray(avoid["expression"], dtype=str)))
            candidates = candidates[~np.isin(np.asarray(candidates["expression"], dtype=str), list(avoid_exprs))]
        except (TypeError, ValueError, IndexError, KeyError):
            pass
        if len(candidates) == 0:
            raise LexicalGapError(error_message or "No %s candidates available after avoid filter" % controlled_pos)
    lemmas = sorted({_source_lemma_for_row(row).lower() for row in candidates})
    lemma = uniform_choice(np.asarray(lemmas, dtype=object))
    lemma_rows = candidates[np.fromiter(
        (_source_lemma_for_row(row).lower() == lemma for row in candidates),
        dtype=bool,
        count=len(candidates),
    )]
    return uniform_choice(lemma_rows)


def choose_matching_row(row,
                        label: str,
                        table,
                        controlled_pos: str,
                        fallback_on_empty: bool=False,
                        avoid=None,
                        error_message: Optional[str]=None,
                        minimum_candidates: Optional[int]=None):
    return choose_row_for_active_zipf(
        get_matches_of(row, label, table),
        controlled_pos,
        fallback_on_empty=fallback_on_empty,
        avoid=avoid,
        error_message=error_message,
        minimum_candidates=minimum_candidates,
    )


def choose_matched_by_row(row,
                          label: str,
                          table,
                          controlled_pos: str,
                          fallback_on_empty: bool=False,
                          avoid=None,
                          error_message: Optional[str]=None,
                          minimum_candidates: Optional[int]=None):
    return choose_row_for_active_zipf(
        get_matched_by(row, label, table),
        controlled_pos,
        fallback_on_empty=fallback_on_empty,
        avoid=avoid,
        error_message=error_message,
        minimum_candidates=minimum_candidates,
    )


def verbs_with_argument_slots(subject_rows, object_rows, verb_space):
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


def _curated_rows_for_expression_families(table,
                                          expressions: Sequence[str],
                                          controlled_pos: str,
                                          expand_inflections: bool=False,
                                          minimum_candidates: int=_CURATED_ZIPF_MIN_CANDIDATES):
    rows = _rows_for_expression_families(
        table,
        expressions,
        expand_inflections=expand_inflections,
    )
    return filter_rows_for_active_zipf(
        rows,
        controlled_pos,
        fallback_on_empty=True,
        minimum_candidates=minimum_candidates,
    )


def _exclude_ing_en_rows(table):
    if len(table) == 0:
        return table
    mask = (np.asarray(table["ing"], dtype=str) != "1") & (np.asarray(table["en"], dtype=str) != "1")
    return table[mask]


def _exclude_ing_surface_rows(table):
    if len(table) == 0:
        return table
    expr = np.asarray(table["expression"], dtype=str)
    return table[~np.char.endswith(expr, "ing")]


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
    return _curated_rows_for_expression_families(
        rows,
        _SUBJECT_RAISING_VERBS,
        "verb",
        expand_inflections=True,
    )


def control_subject_verb_rows():
    rows = verb_rows_for_category("V_control_subj")
    return _curated_rows_for_expression_families(
        rows,
        _CONTROL_SUBJECT_VERBS,
        "verb",
        expand_inflections=True,
    )


def object_raising_verb_rows():
    rows = verb_rows_for_category("V_raising_object")
    return _curated_rows_for_expression_families(
        rows,
        _OBJECT_RAISING_VERBS,
        "verb",
        expand_inflections=True,
    )


def control_object_verb_rows():
    rows = verb_rows_for_category("V_control_object")
    return _curated_rows_for_expression_families(
        rows,
        _CONTROL_OBJECT_VERBS,
        "verb",
        expand_inflections=True,
    )


def subject_raising_adjective_rows():
    """Return only the canonical subject-raising adjectives.

    Filters vocabulary rows by _RAISING_ADJECTIVES so that the curated list
    is the single source of truth for both generators and the overlay pipeline.
    """
    rows = adjective_rows_for_category("Adj_raising_subj")
    return _curated_rows_for_expression_families(rows, _RAISING_ADJECTIVES, "adjective")


def control_subject_adjective_rows():
    rows = adjective_rows_for_category("Adj_control_subj")
    return _curated_rows_for_expression_families(rows, _CONTROL_ADJECTIVES, "adjective")


def tough_adjective_rows():
    rows = adjective_rows_for_category("Adj_tough")
    return _curated_rows_for_expression_families(rows, _TOUGH_ADJECTIVES, "adjective")


def clausal_it_adjective_rows():
    rows = adjective_rows_for_category("Adj_clausal")
    rows = rows[rows["arg_1"] == "expression=it"]
    return _curated_rows_for_expression_families(
        rows,
        _CLAUSAL_IT_ADJECTIVES,
        "adjective",
        minimum_candidates=_CURATED_ZIPF_MIN_CANDIDATES,
    )


def existential_bad_control_subject_verb_rows():
    rows = exclude_source_lemmas(control_subject_verb_rows(), _EXISTENTIAL_CONTROL_SUBJECT_EXCLUDED_VERBS)
    return filter_rows_for_active_zipf(
        rows,
        "verb",
        fallback_on_empty=True,
        minimum_candidates=_CURATED_ZIPF_MIN_CANDIDATES,
    )


def existential_subject_raising_verb_rows():
    rows = verb_rows_for_category("V_raising_subj")
    return _curated_rows_for_expression_families(
        rows,
        _EXISTENTIAL_SUBJECT_RAISING_VERBS,
        "verb",
        expand_inflections=True,
    )


def existential_subject_raising_adjective_rows():
    rows = adjective_rows_for_category("Adj_raising_subj")
    return _curated_rows_for_expression_families(
        rows,
        _EXISTENTIAL_SUBJECT_RAISING_ADJECTIVES,
        "adjective",
    )


@lru_cache(maxsize=1)
def dp_buildable_nominal_rows():
    """Nominals that N_to_DP_mutate can safely realize as DPs."""
    rows = table_union1d(
        table_union1d(
            get_all("category", "N", all_nominals),
            get_all("category", "N/NP", all_nominals),
        ),
        table_union1d(
            get_all("category", "N\\NP[poss]", all_nominals),
            get_all("category", "N/S", all_nominals),
        ),
    )
    return exclude_source_lemmas(rows, _LOW_QUALITY_NOMINAL_EXPRESSIONS)


@lru_cache(maxsize=1)
def simple_common_noun_rows():
    """Plain common N rows for templates that insert raw noun text directly."""
    rows = get_all("category", "N", all_nominals)
    rows = get_all("properNoun", "0", rows)
    rows = table_setdiff1d(rows, get_all("locale", "1", rows))
    return exclude_source_lemmas(rows, _LOW_QUALITY_NOMINAL_EXPRESSIONS)


def _single_token_verb_rows(table):
    if len(table) == 0:
        return table
    expr = np.asarray(table["expression"], dtype=str)
    mask = np.array([" " not in value for value in expr], dtype=bool)
    return table[mask]


def passivizable_participle_rows():
    rows = get_all("passive", "1", get_all("category", "(S\\NP)/NP", get_all("en", "1", get_all("verb", "1"))))
    rows = exclude_source_lemmas(rows, _inventory_core_intr_lemmas())
    rows = exclude_source_lemmas(rows, _PASSIVE_GOOD_VERB_BLOCKLIST)
    rows = exclude_low_quality_overlay_verb_lemmas(rows)
    return _single_token_verb_rows(rows)


def nonpassivizable_participle_rows():
    rows = get_all(
        "passive",
        "0",
        get_all("strict_intrans", "1", get_all("category", "S\\NP", get_all("en", "1", get_all("verb", "1")))),
    )
    en_intransitives = get_all("category", "S\\NP", get_all("en", "1", get_all("verb", "1")))
    curated_rows = _rows_for_expression_families(
        table_setdiff1d(en_intransitives, get_all("passive", "1", en_intransitives)),
        _NONPASSIVIZABLE_PARTICIPLE_VERBS,
        expand_inflections=True,
    )
    rows = table_union1d(rows, curated_rows)
    rows = exclude_transitive_source_lemmas(rows)
    rows = exclude_source_lemmas(rows, _inventory_core_trans_lemmas())
    rows = exclude_source_lemmas(rows, _INTRANSITIVE_CONTRAST_BLOCKLIST)
    rows = exclude_agentive_subject_mismatch_rows(rows)
    rows = exclude_low_quality_overlay_verb_lemmas(rows)
    return _single_token_verb_rows(rows)


def pure_strict_intransitive_rows():
    rows = get_all("strict_intrans", "1", get_all("category", "S\\NP", get_all("verb", "1")))
    rows = exclude_transitive_source_lemmas(rows)
    rows = exclude_source_lemmas(rows, _inventory_core_trans_lemmas())
    rows = exclude_source_lemmas(rows, _INTRANSITIVE_CONTRAST_BLOCKLIST)
    rows = exclude_agentive_subject_mismatch_rows(rows)
    return exclude_low_quality_overlay_verb_lemmas(rows)


def pure_strict_transitive_rows():
    rows = get_all("strict_trans", "1", get_all("category", "(S\\NP)/NP", get_all("verb", "1")))
    rows = exclude_source_lemmas(rows, _inventory_core_intr_lemmas())
    rows = exclude_source_lemmas(rows, _inventory_any_intransitive_lemmas())
    rows = exclude_source_lemmas(rows, _TRANSITIVE_CONTRAST_AMBIGUOUS_BLOCKLIST)
    return exclude_low_quality_overlay_verb_lemmas(rows)


def pure_transitive_rows():
    rows = get_all("category", "(S\\NP)/NP", get_all("verb", "1"))
    rows = exclude_source_lemmas(rows, _inventory_core_intr_lemmas())
    return exclude_low_quality_overlay_verb_lemmas(rows)


def drop_argument_good_verb_rows():
    rows = _rows_for_expression_families(get_all("verb", "1"), _DROP_ARGUMENT_GOOD_VERBS, expand_inflections=True)
    return get_all("strict_trans", "0", rows)


def drop_argument_bad_verb_rows():
    return _rows_for_expression_families(
        get_all("category", "(S\\NP)/NP"),
        _DROP_ARGUMENT_BAD_VERBS,
        expand_inflections=True,
    )


def finite_clause_embedding_verb_rows():
    rows = _curated_rows_for_expression_families(
        get_all("category_2", "V_embedding"),
        _FINITE_CLAUSE_EMBEDDING_VERBS,
        "verb",
        expand_inflections=True,
    )
    return _exclude_ing_surface_rows(_exclude_ing_en_rows(rows))


def tough_vs_raising_2_outer_verb_rows():
    bare_verbs = get_all("bare", "1", get_all("verb", "1"))
    simple_iv = get_all("category", "S\\NP", bare_verbs)
    simple_tv = get_all("category", "(S\\NP)/NP", bare_verbs)
    subj_control = get_all("category", "(S\\NP)/(S[to]\\NP)", bare_verbs)
    obj_control = get_all("category", "((S\\NP)/(S[to]/NP))/NP", bare_verbs)
    infinitival_clause = get_all("category", "(S\\NP)/S[to]", bare_verbs)
    rows = table_union1d(
        table_union1d(simple_iv, simple_tv),
        table_union1d(subj_control, table_union1d(obj_control, infinitival_clause)),
    )
    return table_setdiff1d(rows, get_all("causative", "1", rows))


def inchoative_bad_transitive_rows():
    rows = get_all("category", "(S\\NP)/NP", get_all("verb", "1"))
    # Keep only plain transitive surface forms; PP/particle verbs are excluded
    # structurally rather than by a lexical denylist.
    expr = np.asarray(rows["expression"], dtype=str)
    plain_rows = rows[np.array([" " not in value for value in expr], dtype=bool)]
    alternating_roots = set(map(str, causative_alternating_verb_rows()["root"]))
    keep_mask = ~np.isin(np.asarray(plain_rows["root"], dtype=str), list(alternating_roots))
    return plain_rows[keep_mask]


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


def causative_bad_intransitive_rows():
    rows = get_all("category", "S\\NP", get_all("verb", "1"))
    strict_rows = get_all("strict_intrans", "1", rows)
    extra_rows = _rows_for_expression_families(
        rows,
        _CAUSATIVE_BAD_EXTRA_INTRANSITIVES,
        expand_inflections=True,
    )
    combined = table_union1d(strict_rows, extra_rows)
    alternating_roots = set(map(str, causative_alternating_verb_rows()["root"]))
    transitive_exprs = set(
        map(str, get_all("category", "(S\\NP)/NP", get_all("verb", "1"))["expression"])
    )
    keep_mask = ~np.isin(np.asarray(combined["root"], dtype=str), list(alternating_roots))
    keep_mask &= ~np.isin(np.asarray(combined["expression"], dtype=str), list(transitive_exprs))
    rows = exclude_source_lemmas(combined[keep_mask], _INTRANSITIVE_CONTRAST_BLOCKLIST)
    rows = exclude_agentive_subject_mismatch_rows(rows)
    return exclude_low_quality_overlay_verb_lemmas(rows)


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
    if category_2 == "V_control_subj":
        return _CONTROL_SUBJECT_VERBS
    if category_2 == "V_control_object":
        return _CONTROL_OBJECT_VERBS
    return ()
