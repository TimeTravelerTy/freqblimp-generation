import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import wn
from lemminflect import getInflection, getLemma

from generation_projects.blimp.overlay_guards import curated_template_expressions
from utils.frequency import zipf_for_expression
from utils.vocab_table import BASE_VOCAB_PATH, build_frequency_cache, write_frequency_cache
from utils.data_type import data_type


WORD_RE = re.compile(r"^[a-z]+$")
CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][a-z]+\b")
BIOGRAPHICAL_DATE_RE = re.compile(r"\((?:born in )?\d{3,4}(?:[-\u2013]\d{2,4})?\)")
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_VERB_INVENTORY_PATH = MODULE_DIR / "verb_inventory.json"
LEGACY_VERB_INVENTORY_PATH = MODULE_DIR.parent.parent / "blimp-rare" / "data" / "processed" / "verb_inventory_pruned_particles.json"
DEFAULT_VULGAR_BLOCKLIST_PATH = MODULE_DIR / "profanity_list_en.json"
PROFANITY_TOKEN_RE = re.compile(r"^[a-z*]+$")
NOUN_SUBJECT_TO_BUNDLE = {
    "noun.person": "person",
    "noun.animal": "animal",
    "noun.group": "institution",
    "noun.artifact": "artifact",
    "noun.body": "artifact",
    "noun.object": "artifact",
    "noun.location": "locale",
    "noun.communication": "document",
    "noun.cognition": "conceptual",
    "noun.act": "conceptual",
    "noun.event": "conceptual",
    "noun.state": "conceptual",
    "noun.attribute": "conceptual",
    "noun.food": "food_count",
    "noun.plant": "food_count",
    "noun.substance": "liquid_mass",
}
NOUN_BUNDLE_FIELDS = {
    "person": {"animate": "1", "mass": "0", "properNoun": "0"},
    "artifact": {"animate": "0", "artifact": "1", "physical": "1", "mass": "0", "properNoun": "0"},
    "vehicle": {"animate": "0", "artifact": "1", "vehicle": "1", "physical": "1", "mass": "0", "properNoun": "0"},
    "document": {"animate": "0", "document": "1", "physical": "1", "mass": "0", "properNoun": "0"},
    "conceptual": {"animate": "0", "conceptual": "1", "mass": "0", "properNoun": "0"},
    "food_count": {"animate": "0", "food": "1", "physical": "1", "mass": "0", "properNoun": "0"},
    "liquid_mass": {"animate": "0", "liquid": "1", "physical": "1", "mass": "1", "properNoun": "0"},
    "animal": {"animal": "1", "mass": "0", "properNoun": "0"},
    "locale": {"locale": "1", "institution": "", "mass": "0", "properNoun": "0"},
    "institution": {"institution": "1", "locale": "", "mass": "0", "properNoun": "0"},
    "locale_institution": {"locale": "1", "institution": "1", "mass": "0", "properNoun": "0"},
}
NOUN_OUTPUT_SEMANTIC_FIELDS = (
    "agent",
    "animal",
    "animate",
    "appearance",
    "artifact",
    "clothing",
    "conceptual",
    "document",
    "food",
    "institution",
    "liquid",
    "locale",
    "occupation",
    "person",
    "physical",
    "properNoun",
    "vehicle",
)
NOUN_BUNDLE_OUTPUT_FIELDS = {
    "person": {"animate": "1", "person": "1", "physical": "1", "agent": "1", "mass": "0", "properNoun": "0"},
    "artifact": {"animate": "0", "artifact": "1", "physical": "1", "mass": "0", "properNoun": "0"},
    "vehicle": {"animate": "0", "artifact": "1", "vehicle": "1", "physical": "1", "mass": "0", "properNoun": "0"},
    "document": {"animate": "0", "document": "1", "physical": "1", "mass": "0", "properNoun": "0"},
    "conceptual": {"animate": "0", "conceptual": "1", "mass": "0", "properNoun": "0"},
    "food_count": {"animate": "0", "food": "1", "physical": "1", "mass": "0", "properNoun": "0"},
    "liquid_mass": {"animate": "0", "liquid": "1", "physical": "1", "mass": "1", "properNoun": "0"},
    "animal": {"animal": "1", "animate": "1", "physical": "1", "agent": "1", "mass": "0", "properNoun": "0"},
    "locale": {"locale": "1", "institution": "", "mass": "0", "properNoun": "0"},
    "institution": {"institution": "1", "animate": "0", "mass": "0", "properNoun": "0"},
    "locale_institution": {"locale": "1", "institution": "1", "animate": "0", "mass": "0", "properNoun": "0"},
}
VERB_SIG_FIELDS = (
    "category",
    "category_2",
    "arg_1",
    "arg_2",
    "strict_intrans",
    "strict_trans",
    "causative",
    "inchoative",
    "passive",
)
FIELD_ALIASES = {
    "synonym/hypernym/hyponym": "synonym_hypernym_hyponym",
}
ADJECTIVE_TEMPLATE_FIELDS = {
    "animate": "animate=1",
    "artifact": "artifact=1",
    "cleanable": "cleanable=1",
    "document": "document=1",
    "generic": "category=N^frequent=1",
    "physical": "physical=1",
    "physical_inanimate": "physical=1^animate=0",
    "vegetable": "vegetable=1",
}
SPECIAL_ADJECTIVE_CATEGORY2 = (
    "Adj_clausal",
    "Adj_control_subj",
    "Adj_raising_subj",
    "Adj_tough",
)
SPECIAL_VERB_CATEGORY2 = (
    "V_control_object",
    "V_control_subj",
    "V_raising_object",
    "V_raising_subj",
)
VERB_FRAME_OVERRIDES = {
    # Keep these corrections in the tracked overlay builder because
    # verb_inventory.json is a large ignored artifact in this repo.
    "go": {"remove": {"trans"}, "add": set()},
    "bulge": {"remove": {"trans"}, "add": set()},
    "cascade": {"remove": {"trans"}, "add": set()},
    "guzzle": {"remove": set(), "add": {"intr"}},
    "redecorate": {"remove": set(), "add": {"intr"}},
}
HEAD_AMBIGUOUS_INTRANSITIVE_CLASSES = {
    # High-frequency denominal/deadjectival uses whose WordNet verb sense is
    # real but too fringe for simple BLiMP intransitive templates.
    "antique": "denominal_object",
    "bucket": "denominal_object",
    "charcoal": "deadjectival",
    "hay": "denominal_object",
    "holiday": "denominal_time",
    "honeymoon": "denominal_time",
    "low": "deadjectival",
    "nut": "denominal_object",
    "pearl": "denominal_object",
    "summer": "denominal_time",
    "twitter": "entity_like",
    "vacation": "denominal_time",
    "war": "denominal_event",
    "weekend": "denominal_time",
    "winter": "denominal_time",
}
LOW_QUALITY_NOUN_EXPRESSIONS = frozenset({
    # Nominalized adjectives that are technically noun lemmas but unstable as
    # arbitrary count nouns in BLiMP minimal-pair templates.
    "airs",
    "altogether",
    "colonial",
    "contemporary",
    "dependent",
    "dining",
    "disabled",
    "divine",
    "drinkable",
    "empty",
    "familiar",
    "federal",
    "favorite",
    "favourite",
    "homosexual",
    "immune",
    "independent",
    "invalid",
    "moderate",
    "neutral",
    "poor",
    "prior",
    "probable",
    "posing",
    "regular",
    "rich",
    "romantic",
    "semitic",
    "silly",
    "skating",
    "steady",
    "temporary",
    "wounded",
})
ALLOWED_ING_NOUN_EXPRESSIONS = frozenset({
    "building",
    "drawing",
    "offspring",
    "painting",
})
AGENTIVE_VERB_SUBJECT_DOMAINS = frozenset({
    "verb.body",
    "verb.cognition",
    "verb.communication",
    "verb.competition",
    "verb.consumption",
    "verb.emotion",
    "verb.possession",
    "verb.social",
})
AGENTIVE_ARG_MARKERS = (
    "animal=1",
    "animate=1",
    "institution=1",
    "person=1",
)
ADJECTIVE_RELATIONAL_PREFIXES = (
    "of or relating to ",
    "relating to ",
    "of or characteristic of ",
    "characteristic of ",
    "being of or having to do with ",
    "of or belonging to ",
    "of the nature of ",
    "pertaining to ",
)
ADJECTIVE_BLOCKLIST = {
    "another",
    "more",
    "none",
    "undefined",
}
ROMAN_NUMERAL_RE = re.compile(r"^(?=[ivxlcdm]+$)[ivxlcdm]+$")
NOUN_ENTITY_MARKERS = (
    "continent",
    "goddess",
    "ocean",
    "river",
    "state",
)
NOUN_LANGUAGE_MARKERS = (
    " dialect ",
    " language ",
)
ADJECTIVE_VEGETABLE_MARKERS = (
    "vegetable",
    "vegetables",
    "fruit",
    "fruits",
    "crop",
    "crops",
    "harvest",
    "harvested",
    "ripe",
    "unripe",
)
ADJECTIVE_DOCUMENT_MARKERS = (
    "document",
    "documents",
    "book",
    "books",
    "paper",
    "papers",
    "report",
    "reports",
    "text",
    "texts",
    "writing",
    "written",
    "publication",
    "publications",
    "story",
    "stories",
)
ADJECTIVE_CLEANABLE_MARKERS = (
    "clean",
    "cleaned",
    "dirty",
    "messy",
    "mud",
    "muck",
    "stain",
    "stained",
    "wash",
    "washed",
)
ADJECTIVE_ANIMATE_MARKERS = (
    "living thing",
    "living things",
    "alive",
    "person",
    "persons",
    "people",
    "human",
    "humans",
    "animal",
    "animals",
    "creature",
    "creatures",
    "organism",
    "organisms",
    "eat food",
    "hunger",
    "hungry",
    "sleep",
    "asleep",
    "youth",
)
ADJECTIVE_ARTIFACT_MARKERS = (
    "artifact",
    "artifacts",
    "device",
    "devices",
    "machine",
    "machines",
    "tool",
    "tools",
    "vehicle",
    "vehicles",
    "furniture",
    "clothing",
    "garment",
    "garments",
)
ADJECTIVE_PHYSICAL_INANIMATE_MARKERS = (
    "of the color",
    "color",
    "coloured",
    "colored",
    "material",
    "made or consisting",
    "made of",
    "consisting of",
    "covered with",
    "of soil",
    "of liquids",
    "watery",
    "sediment",
    "wood",
    "wooden",
    "metal",
    "metallic",
    "glass",
    "mud",
)
ADJECTIVE_PHYSICAL_MARKERS = (
    "physical",
    "physically",
    "visible",
    "concealed",
    "exposed",
    "open",
    "closed",
    "surface",
    "space",
    "shape",
    "size",
    "length",
    "long",
    "short",
    "big",
    "small",
    "solid",
    "piece",
    "pieces",
)


def _record_rejection(audit, overlay_type, lemma, reason, **details):
    audit["summary"]["rejections"][overlay_type][reason] += 1
    entry = {"overlay_type": overlay_type, "lemma": lemma, "reason": reason}
    for key, value in details.items():
        if value is not None:
            entry[key] = value
    audit["rejected"].append(entry)


def _record_admission(audit, overlay_type, lemma, zipf, inherited_template, bundle, row_count):
    audit["summary"]["admitted"][overlay_type] += 1
    entry = {
        "overlay_type": overlay_type,
        "lemma": lemma,
        "zipf": zipf,
        "inherited_template": inherited_template,
        "bundle": bundle,
        "row_count": row_count,
    }
    audit["admitted"].append(entry)


def _freeze_summary(audit):
    return {
        "admitted": dict(audit["summary"]["admitted"]),
        "rejections": {
            overlay_type: dict(reason_counts)
            for overlay_type, reason_counts in audit["summary"]["rejections"].items()
        },
    }


def _load_rows(path):
    with open(path, newline="") as handle:
        rows = []
        for raw_row in csv.DictReader(handle):
            row = dict(raw_row)
            for target, source in FIELD_ALIASES.items():
                if target not in row and source in row:
                    row[target] = row[source]
            rows.append(row)
        return rows


def _write_rows(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_value(row, field):
    if field in row:
        return row[field]
    alias = FIELD_ALIASES.get(field)
    if alias and alias in row:
        return row[alias]
    return ""


def _row_signature_from_dict(row):
    return "||".join("%s=%s" % (field, _row_value(row, field)) for field, _dtype in data_type)


def _safe_root_token(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


def _empty_blocklist():
    return {"entries": []}


def _compile_repeated_char_fragment(pattern):
    pattern = str(pattern).strip().lower()
    if not pattern or not PROFANITY_TOKEN_RE.fullmatch(pattern):
        return None
    pieces = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            return None
        repeated = index + 1 < len(pattern) and pattern[index + 1] == "*"
        pieces.append(re.escape(char) + ("+" if repeated else ""))
        index += 2 if repeated else 1
    return "".join(pieces)


def _compile_exception_pattern(template, match_fragment):
    template = str(template).strip().lower()
    if not template or not PROFANITY_TOKEN_RE.fullmatch(template):
        return None
    pieces = []
    for char in template:
        if char == "*":
            pieces.append("(?:%s)" % match_fragment)
        else:
            pieces.append(re.escape(char))
    return re.compile("^%s$" % "".join(pieces))


def _append_matcher(
    entries,
    seen,
    regex,
    match_value,
    profanity_id=None,
    severity=None,
    tags=None,
    exception_regexes=None,
    priority=0,
):
    if regex is None:
        return
    key = (match_value, profanity_id)
    if key in seen:
        return
    seen.add(key)
    entries.append({
        "regex": regex,
        "match_value": match_value,
        "profanity_id": profanity_id,
        "severity": severity,
        "tags": tuple(tags or ()),
        "exception_regexes": tuple(exception_regexes or ()),
        "priority": priority,
    })


def _extend_profanity_list(entries, seen, payload):
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        match_value = str(entry.get("match", "")).strip().lower()
        if not match_value:
            continue
        profanity_id = str(entry.get("id", "")).strip().lower() or None
        try:
            severity = int(entry.get("severity")) if entry.get("severity") is not None else None
        except (TypeError, ValueError):
            severity = None
        tags = tuple(sorted(str(tag).strip().lower() for tag in entry.get("tags", []) if str(tag).strip()))
        raw_exceptions = entry.get("exceptions", [])
        for variant in match_value.split("|"):
            variant = variant.strip().lower()
            if not variant or not PROFANITY_TOKEN_RE.fullmatch(variant):
                continue
            match_fragment = _compile_repeated_char_fragment(variant)
            if match_fragment is None:
                continue
            regex = re.compile("^%s$" % match_fragment)
            exception_regexes = []
            for exception in raw_exceptions:
                compiled = _compile_exception_pattern(exception, match_fragment)
                if compiled is not None:
                    exception_regexes.append(compiled)
            _append_matcher(
                entries,
                seen,
                regex,
                variant,
                profanity_id=profanity_id,
                severity=severity,
                tags=tags,
                exception_regexes=exception_regexes,
                priority=4,
            )


def _load_json_payload(path):
    candidate_path = Path(path)
    if not candidate_path.exists():
        raise FileNotFoundError("Missing profanity list: %s" % candidate_path)
    return json.loads(candidate_path.read_text())


def _load_vulgar_blocklist(path, supplement_path=None):
    entries = []
    seen = set()
    if path:
        payload = _load_json_payload(path)
        if not isinstance(payload, list):
            raise ValueError("Profanity list must be a JSON array of objects: %s" % path)
        _extend_profanity_list(entries, seen, payload)
    if supplement_path and Path(supplement_path) != Path(path or ""):
        payload = _load_json_payload(supplement_path)
        if not isinstance(payload, list):
            raise ValueError("Profanity list must be a JSON array of objects: %s" % supplement_path)
        _extend_profanity_list(entries, seen, payload)
    entries.sort(key=lambda entry: (-entry["priority"], -(entry["severity"] or 0), -len(entry["match_value"])))
    return {"entries": entries}


def _match_blocked_lemma(lemma, blocklist):
    lemma = str(lemma).strip().lower()
    if not lemma:
        return None
    for entry in blocklist.get("entries", ()):
        if entry["regex"].fullmatch(lemma):
            if any(regex.fullmatch(lemma) for regex in entry.get("exception_regexes", ())):
                continue
            match = {"match_value": entry["match_value"]}
            if entry.get("profanity_id"):
                match["profanity_id"] = entry["profanity_id"]
            if entry.get("severity") is not None:
                match["severity"] = entry["severity"]
            if entry.get("tags"):
                match["tags"] = list(entry["tags"])
            return match
    return None


def _load_oe_lemmas(pos, lexicon):
    lex = wn.Wordnet(lexicon)
    entries = lex.words(pos=pos)
    seen = set()
    lemmas = []
    for entry in entries:
        lemma = entry.lemma().lower()
        if lemma in seen or not WORD_RE.fullmatch(lemma):
            continue
        seen.add(lemma)
        lemmas.append(lemma)
    return lemmas


def _load_oe_adjective_lemmas(lexicon):
    seen = set()
    lemmas = []
    for pos in ("a", "s"):
        for lemma in _load_oe_lemmas(pos, lexicon):
            if lemma in seen:
                continue
            seen.add(lemma)
            lemmas.append(lemma)
    return lemmas


def _noun_subject(lemma, lexicon):
    try:
        synsets = wn.synsets(lemma, pos="n", lexicon=lexicon)
    except Exception:
        return []
    if not synsets:
        return []
    subjects = []
    for synset in synsets:
        metadata = synset.metadata() if callable(getattr(synset, "metadata", None)) else {}
        subject = metadata.get("subject") if isinstance(metadata, dict) else None
        if subject == "noun.artifact":
            definition = (synset.definition() or "").lower()
            if "vehicle" in definition:
                subject = "noun.vehicle"
        if subject:
            subjects.append(subject)
    return subjects


def _noun_bundle_for_lemma(lemma, lexicon):
    subjects = _noun_subject(lemma, lexicon)
    if not subjects:
        return None, 0.0, []
    bundle_counts = defaultdict(int)
    for subject in subjects:
        if subject == "noun.vehicle":
            bundle_counts["vehicle"] += 1
            continue
        bundle = NOUN_SUBJECT_TO_BUNDLE.get(subject)
        if bundle:
            bundle_counts[bundle] += 1
    if not bundle_counts:
        return None, 0.0, subjects
    bundle, count = max(bundle_counts.items(), key=lambda item: (item[1], item[0]))
    # Count unsupported senses against confidence so a single mapped fringe sense
    # does not dominate a polysemous noun like "dimple" or "takedown".
    confidence = count / len(subjects)
    return bundle, confidence, subjects


def _noun_expression_looks_plural(lemma):
    lemmas = tuple(str(value).strip().lower() for value in (getLemma(lemma, upos="NOUN") or ()))
    if not lemmas:
        return False
    return all(value and value != lemma for value in lemmas)


def _noun_entity_like(lemma, lexicon):
    try:
        synsets = wn.synsets(lemma, pos="n", lexicon=lexicon)
    except Exception:
        return False, []
    definitions = [(synset.definition() or "").strip() for synset in synsets]
    if not definitions:
        return False, []
    lowered = [" %s " % re.sub(r"[^a-z]+", " ", definition.lower()).strip() for definition in definitions]
    if all(_definition_has_name_like_markers(definition) for definition in definitions):
        return True, definitions
    if any(any(" %s " % marker in definition for marker in NOUN_ENTITY_MARKERS) for definition in lowered):
        return True, definitions
    if any(any(marker in definition for marker in NOUN_LANGUAGE_MARKERS) for definition in lowered):
        try:
            if wn.synsets(lemma, pos="a", lexicon=lexicon) or wn.synsets(lemma, pos="s", lexicon=lexicon):
                return True, definitions
        except Exception:
            return True, definitions
    return False, definitions


def _definition_has_name_like_markers(definition):
    definition = str(definition or "")
    if not definition:
        return False
    lowered = definition.lower()
    if BIOGRAPHICAL_DATE_RE.search(definition):
        return True
    if "born in " in lowered:
        return True
    return bool(CAPITALIZED_TOKEN_RE.search(definition))


def _synsets_for(lemma, pos, lexicon):
    try:
        return wn.synsets(lemma, pos=pos, lexicon=lexicon)
    except Exception:
        return []


def _definitions_for(synsets):
    return [(synset.definition() or "").strip() for synset in synsets if synset.definition()]


def _verb_subject_domains(lemma, lexicon):
    domains = set()
    for synset in _synsets_for(lemma, "v", lexicon):
        metadata = synset.metadata() if callable(getattr(synset, "metadata", None)) else {}
        subject = metadata.get("subject") if isinstance(metadata, dict) else None
        if subject:
            domains.add(subject)
    return frozenset(domains)


def _verb_requires_agentive_subject(lemma, lexicon):
    domains = _verb_subject_domains(lemma, lexicon)
    return bool(domains) and domains.issubset(AGENTIVE_VERB_SUBJECT_DOMAINS)


def _row_allows_agentive_subject(row):
    requirement = str(row.get("arg_1", ""))
    return any(marker in requirement for marker in AGENTIVE_ARG_MARKERS)


def _filter_verb_rows_for_subject_quality(rows, lemma, lexicon):
    if not _verb_requires_agentive_subject(lemma, lexicon):
        return rows
    return [row for row in rows if _row_allows_agentive_subject(row)]


def _head_ambiguous_intransitive_rejection(lemma, entry, zipf_value, lexicon):
    frame_types = entry.get("frame_types", set())
    if zipf_value < 3.5 or "intr" not in frame_types:
        return None, {}

    verb_synsets = _synsets_for(lemma, "v", lexicon)
    noun_synsets = _synsets_for(lemma, "n", lexicon)
    adjective_synsets = _synsets_for(lemma, "a", lexicon) + _synsets_for(lemma, "s", lexicon)
    verb_definitions = _definitions_for(verb_synsets)

    explicit_class = HEAD_AMBIGUOUS_INTRANSITIVE_CLASSES.get(lemma)
    if explicit_class:
        return "head_ambiguous_intransitive", {
            "noise_class": explicit_class,
            "verb_senses": len(verb_synsets),
            "noun_senses": len(noun_synsets),
            "adjective_senses": len(adjective_synsets),
            "definitions": verb_definitions[:3],
        }

    if "trans" in frame_types:
        return None, {}

    if len(noun_synsets) + len(adjective_synsets) < len(verb_synsets):
        return None, {}

    lowered = [" %s " % re.sub(r"[^a-z]+", " ", definition.lower()).strip() for definition in verb_definitions]
    if any(" spend the " in definition or " spend or take " in definition for definition in lowered):
        noise_class = "denominal_time"
    elif any(" gather " in definition or " harvest " in definition for definition in lowered):
        noise_class = "denominal_object"
    elif any(" put " in definition and " into " in definition for definition in lowered):
        noise_class = "denominal_container"
    elif any(" travel by " in definition or " ride in " in definition for definition in lowered):
        noise_class = "denominal_vehicle"
    elif adjective_synsets and len(verb_synsets) <= 1:
        noise_class = "deadjectival"
    elif verb_definitions and all(_definition_has_name_like_markers(definition) for definition in verb_definitions):
        noise_class = "entity_like"
    else:
        return None, {}

    return "head_ambiguous_intransitive", {
        "noise_class": noise_class,
        "verb_senses": len(verb_synsets),
        "noun_senses": len(noun_synsets),
        "adjective_senses": len(adjective_synsets),
        "definitions": verb_definitions[:3],
    }


def _low_quality_noun_rejection(lemma):
    if lemma in LOW_QUALITY_NOUN_EXPRESSIONS:
        return "low_quality_nominalized_adjective"
    if lemma.endswith("ing") and lemma not in ALLOWED_ING_NOUN_EXPRESSIONS:
        verb_lemmas = {str(value).strip().lower() for value in (getLemma(lemma, upos="VERB") or ())}
        if any(value and value != lemma for value in verb_lemmas):
            return "deverbal_gerund_noun"
    return None


def _noun_person_name_like(lemma, lexicon):
    try:
        synsets = wn.synsets(lemma, pos="n", lexicon=lexicon)
    except Exception:
        return False, []
    person_definitions = []
    generic_definition_found = False
    for synset in synsets:
        metadata = synset.metadata() if callable(getattr(synset, "metadata", None)) else {}
        subject = metadata.get("subject") if isinstance(metadata, dict) else None
        if subject != "noun.person":
            continue
        definition = synset.definition() or ""
        person_definitions.append(definition)
        if not _definition_has_name_like_markers(definition):
            generic_definition_found = True
    if not person_definitions:
        return False, []
    return (not generic_definition_found), person_definitions


def _noun_template_index(rows):
    index = defaultdict(list)
    for row in rows:
        if row["category"] != "N" or row["properNoun"] != "0":
            continue
        for bundle, fields in NOUN_BUNDLE_FIELDS.items():
            if all(row.get(key, "") == value for key, value in fields.items()):
                index[bundle].append(row)
    return index


def _adjective_template_index(rows):
    index = defaultdict(list)
    arg_to_bundle = {arg_1: bundle for bundle, arg_1 in ADJECTIVE_TEMPLATE_FIELDS.items()}
    for row in rows:
        if row["category"] != "N/N" or row["non_v_pred"] != "1":
            continue
        if row["category_2"] not in {"Adj", "adjective"}:
            continue
        bundle = arg_to_bundle.get(row["arg_1"])
        if bundle is not None:
            index[bundle].append(row)
    return index


def _special_adjective_template_index(rows):
    index = defaultdict(list)
    for row in rows:
        category_2 = row.get("category_2", "")
        if category_2 not in SPECIAL_ADJECTIVE_CATEGORY2:
            continue
        index[category_2].append(row)
    for category_2, family in index.items():
        preferred = {expr.lower() for expr in curated_template_expressions(category_2)}
        family.sort(
            key=lambda row: (
                0 if str(row["expression"]).strip().lower() in preferred else 1,
                str(row["expression"]),
            )
        )
    return index


def _choose_template(rows):
    singular = [row for row in rows if row.get("sg", "") == "1"]
    singular.sort(key=lambda row: row["expression"])
    if singular:
        return singular[0]
    rows = sorted(rows, key=lambda row: row["expression"])
    if not rows:
        return None
    return rows[0]


def _pluralize_noun(lemma):
    forms = getInflection(lemma, tag="NNS")
    return forms[0].lower() if forms else None


def _starts_with_vowel(text):
    text = str(text or "").strip().lower()
    return "1" if text[:1] in {"a", "e", "i", "o", "u"} else "0"


def _normalize_noun_row(row, expression, singularform="", pluralform=""):
    row["expression"] = expression
    row["singularform"] = singularform
    row["pluralform"] = pluralform
    row["frequent"] = "1"
    row["irrpl"] = ""
    row["sgequalspl"] = ""
    row["homophonous"] = ""
    row["start_with_vowel"] = _starts_with_vowel(expression)
    return row


def _apply_noun_bundle_output_fields(row, bundle):
    for field in NOUN_OUTPUT_SEMANTIC_FIELDS:
        if field in row:
            row[field] = ""
    for field, value in NOUN_BUNDLE_OUTPUT_FIELDS.get(bundle, {}).items():
        row[field] = value
    return row


def _normalize_adjective_row(row, expression):
    row["expression"] = expression
    row["frequent"] = "1"
    return row


def _make_noun_rows(lemma, template_row, bundle):
    plural = _pluralize_noun(lemma)
    if template_row is None:
        return []
    if template_row.get("mass") == "1":
        singular_row = dict(template_row)
        _normalize_noun_row(singular_row, lemma)
        _apply_noun_bundle_output_fields(singular_row, bundle)
        return [singular_row]
    if not plural or plural == lemma:
        return []
    singular_row = dict(template_row)
    plural_candidates = [
        row for row in _BASE_ROWS
        if row["category"] == "N"
        and row["sg"] == "0"
        and row["pl"] == "1"
        and row.get("singularform", "") == template_row["expression"]
    ]
    if not plural_candidates:
        return []
    plural_row = dict(plural_candidates[0])
    _normalize_noun_row(singular_row, lemma, pluralform=plural)
    _normalize_noun_row(plural_row, plural, singularform=lemma)
    _apply_noun_bundle_output_fields(singular_row, bundle)
    _apply_noun_bundle_output_fields(plural_row, bundle)
    return [singular_row, plural_row]


def _adjective_synsets(lemma, lexicon):
    synsets = []
    seen = set()
    for pos in ("a", "s"):
        try:
            for synset in wn.synsets(lemma, pos=pos, lexicon=lexicon):
                if synset.id in seen:
                    continue
                seen.add(synset.id)
                synsets.append(synset)
        except Exception:
            continue
    return synsets


def _definition_contains_any(definition, markers):
    normalized = " %s " % re.sub(r"[^a-z]+", " ", definition.lower()).strip()
    for marker in markers:
        marker = str(marker).strip().lower()
        if not marker:
            continue
        if " " in marker:
            if marker in definition:
                return True
            continue
        if " %s " % marker in normalized:
            return True
    return False


def _adjective_definition_bundle(definition):
    definition = str(definition or "").strip().lower()
    if not definition:
        return None
    if definition.startswith(ADJECTIVE_RELATIONAL_PREFIXES):
        return None
    if _definition_contains_any(definition, ADJECTIVE_VEGETABLE_MARKERS):
        return "vegetable"
    if _definition_contains_any(definition, ADJECTIVE_DOCUMENT_MARKERS):
        return "document"
    if _definition_contains_any(definition, ADJECTIVE_CLEANABLE_MARKERS):
        return "cleanable"
    if _definition_contains_any(definition, ADJECTIVE_ANIMATE_MARKERS):
        return "animate"
    if _definition_contains_any(definition, ADJECTIVE_ARTIFACT_MARKERS):
        return "artifact"
    if _definition_contains_any(definition, ADJECTIVE_PHYSICAL_INANIMATE_MARKERS):
        return "physical_inanimate"
    if _definition_contains_any(definition, ADJECTIVE_PHYSICAL_MARKERS):
        return "physical"
    return "generic"


def _adjective_bundle_for_lemma(lemma, lexicon):
    if lemma in ADJECTIVE_BLOCKLIST or ROMAN_NUMERAL_RE.fullmatch(lemma):
        return None, 0.0, []
    synsets = _adjective_synsets(lemma, lexicon)
    if not synsets:
        return None, 0.0, []
    definitions = [(synset.definition() or "").strip() for synset in synsets]
    bundle_counts = defaultdict(int)
    non_relational = 0
    relational_only = True
    for synset in synsets:
        definition = (synset.definition() or "").strip()
        metadata = synset.metadata() if callable(getattr(synset, "metadata", None)) else {}
        subject = metadata.get("subject") if isinstance(metadata, dict) else None
        if subject != "adj.pert" and not definition.lower().startswith(ADJECTIVE_RELATIONAL_PREFIXES):
            relational_only = False
        bundle = _adjective_definition_bundle(definition)
        if bundle is None:
            continue
        non_relational += 1
        bundle_counts[bundle] += 1
    if relational_only:
        return None, 0.0, definitions
    if not bundle_counts:
        return None, 0.0, definitions
    bundle, count = max(bundle_counts.items(), key=lambda item: (item[1], item[0]))
    denominator = non_relational or len(synsets)
    return bundle, count / denominator, definitions


def _make_adjective_rows(lemma, template_row):
    if template_row is None:
        return []
    row = dict(template_row)
    _normalize_adjective_row(row, lemma)
    return [row]


def _load_verb_inventory(path):
    payload = json.loads(Path(path).read_text())
    entries = []
    for entry in payload:
        lemma = str(entry.get("lemma", "")).strip().lower()
        if not lemma or not WORD_RE.fullmatch(lemma):
            continue
        frame_values = defaultdict(set)
        frame_types = {frame.get("type") for frame in entry.get("frames", []) if isinstance(frame, dict)}
        override = VERB_FRAME_OVERRIDES.get(lemma)
        if override is not None:
            frame_types.difference_update(override["remove"])
            frame_types.update(override["add"])
        for frame in entry.get("frames", []):
            if not isinstance(frame, dict):
                continue
            frame_type = frame.get("type")
            if frame_type in {"intr_pp", "ditrans_pp"} and frame.get("prep"):
                frame_values[frame_type].add(str(frame["prep"]).strip().lower())
            if frame_type in {"intr_particle", "trans_particle", "ditrans_particle"} and frame.get("particle"):
                frame_values[frame_type].add(str(frame["particle"]).strip().lower())
        entries.append({"lemma": lemma, "frame_types": frame_types})
        entries[-1]["frame_values"] = {key: tuple(sorted(values)) for key, values in frame_values.items()}
    return entries


def _core_template_keys(frame_types):
    has_intr = "intr" in frame_types
    has_trans = "trans" in frame_types
    keys = []
    if has_intr:
        keys.append("intr")
    if has_trans:
        keys.append("trans")
    return tuple(keys)


def _core_verb_template_index(rows):
    index = {"intr": [], "trans": []}
    by_root = defaultdict(list)
    for row in rows:
        if row["verb"] == "1" and row["category"] in {"S\\NP", "(S\\NP)/NP"}:
            by_root[row["root"]].append(row)
    for root, family in by_root.items():
        if any("_" in row["expression"] or " " in row["expression"] or "-" in row["expression"] for row in family):
            continue
        category = family[0]["category"]
        key = "intr" if category == "S\\NP" else "trans"
        index[key].append(sorted(family, key=lambda row: row["expression"]))
    for key in index:
        index[key].sort(key=lambda family: family[0]["root"])
    return index


def _special_verb_template_index(rows):
    index = defaultdict(list)
    by_root = defaultdict(list)
    for row in rows:
        if row["verb"] != "1":
            continue
        category_2 = row.get("category_2", "")
        if category_2 not in SPECIAL_VERB_CATEGORY2:
            continue
        by_root[(category_2, row["root"])].append(row)
    for (category_2, _root), family in by_root.items():
        family = sorted(family, key=lambda row: row["expression"])
        if any("_" in row["expression"] or "-" in row["expression"] for row in family):
            continue
        suffix_token = _template_suffix_token(family) if any(" " in row["expression"] for row in family) else None
        if any(" " in row["expression"] for row in family) and suffix_token is None:
            continue
        index[category_2].append((family, suffix_token))
    for category_2, families in index.items():
        preferred = {expr.lower() for expr in curated_template_expressions(category_2)}
        families.sort(
            key=lambda item: (
                0 if str(item[0][0]["expression"]).strip().lower() in preferred else 1,
                str(item[0][0]["root"]),
            )
        )
    return index


def _verb_sig_key(row):
    return tuple(row.get(field, "") for field in VERB_SIG_FIELDS)


def _sig_verb_template_index(coarse_index):
    sig_index = {}
    for coarse_key, families in coarse_index.items():
        by_sig = {}
        for family in families:
            sig_key = _verb_sig_key(family[0])
            if sig_key not in by_sig:
                by_sig[sig_key] = family
        sig_index[coarse_key] = by_sig
    return sig_index


def _template_suffix_token(template_family):
    bare_rows = [row for row in template_family if row["bare"] == "1"]
    template_row = bare_rows[0] if bare_rows else template_family[0]
    tokens = template_row["expression"].strip().split()
    if len(tokens) != 2:
        return None
    return tokens[-1].lower()


def _special_verb_candidate_keys(entry):
    lemma = str(entry.get("lemma", "")).strip().lower()
    frame_types = set(entry.get("frame_types") or ())
    keys = []
    for category_2 in SPECIAL_VERB_CATEGORY2:
        if lemma not in set(curated_template_expressions(category_2)):
            continue
        if category_2 in {"V_raising_subj", "V_control_subj"} and ("intr" in frame_types or "trans" in frame_types):
            keys.append(category_2)
        elif category_2 in {"V_raising_object", "V_control_object"} and "trans" in frame_types:
            keys.append(category_2)
    return keys


def _multiword_verb_template_index(rows, category):
    index = defaultdict(list)
    by_root = defaultdict(list)
    for row in rows:
        if row["verb"] == "1" and row["category"] == category and " " in row["expression"]:
            by_root[row["root"]].append(row)
    for root, family in by_root.items():
        suffix_token = _template_suffix_token(family)
        if not suffix_token:
            continue
        index[suffix_token].append(sorted(family, key=lambda row: row["expression"]))
    for suffix_token in index:
        index[suffix_token].sort(key=lambda family: family[0]["root"])
    return index


def _verb_form_for_row(lemma, row):
    if row["bare"] == "1":
        tag = "VB"
    elif row["ing"] == "1":
        tag = "VBG"
    elif row["en"] == "1":
        tag = "VBN"
    elif row["past"] == "1":
        tag = "VBD"
    elif row["pres"] == "1" and row["3sg"] == "1":
        tag = "VBZ"
    elif row["pres"] == "1":
        tag = "VBP"
    else:
        return None
    forms = getInflection(lemma, tag=tag)
    if not forms:
        return None
    return forms[0].lower()


def _make_verb_rows(lemma, template_family, template_label, suffix_token=None):
    category = template_family[0]["category"]
    root_suffix = _safe_root_token(category)
    new_root = "%s_overlay_%s_%s" % (lemma, root_suffix, _safe_root_token(template_label))
    rows = []
    for template_row in template_family:
        form = _verb_form_for_row(lemma, template_row)
        if not form:
            return []
        row = dict(template_row)
        row["expression"] = "%s %s" % (form, suffix_token) if suffix_token else form
        row["root"] = new_root
        row["frequent"] = "1"
        row["irr_verb"] = ""
        row["irr_past"] = ""
        row["special_en_form"] = ""
        row["homophonous"] = ""
        rows.append(row)
    return rows


def _apply_argument_structure_overrides(rows, entry):
    frame_types = set(entry.get("frame_types") or ())
    has_intr = "intr" in frame_types
    has_trans = "trans" in frame_types
    if not has_intr and not has_trans:
        return rows
    for row in rows:
        if row.get("verb") != "1":
            continue
        category = row.get("category")
        if category == "S\\NP" and has_intr:
            row["causative"] = "1"
            if row.get("inchoative", "") == "":
                row["inchoative"] = "0"
        elif category == "(S\\NP)/NP" and has_trans:
            row["inchoative"] = "1"
            if row.get("causative", "") == "":
                row["causative"] = "0"
    return rows


def _base_runtime_table(base_rows, overlay_rows):
    import numpy as np
    return np.array(
        [tuple(_row_value(row, field) for field, _dtype in data_type) for row in base_rows + overlay_rows],
        dtype=data_type,
    )


def build_overlay(args):
    rng = random.Random(args.seed)
    vulgar_blocklist = _empty_blocklist() if args.disable_vulgar_blocklist else _load_vulgar_blocklist(
        args.vulgar_blocklist_path,
    )
    existing_expressions = {row["expression"] for row in _BASE_ROWS}
    existing_signatures = {_row_signature_from_dict(row) for row in _BASE_ROWS}
    overlay_rows = []
    manifest_rows = []
    audit = {
        "admitted": [],
        "rejected": [],
        "summary": {
            "admitted": defaultdict(int),
            "rejections": defaultdict(lambda: defaultdict(int)),
        },
    }

    if args.include_nouns:
        noun_templates = _noun_template_index(_BASE_ROWS)
        noun_candidates = _load_oe_lemmas("n", args.lexicon)
        rng.shuffle(noun_candidates)
        admitted = 0
        for lemma in noun_candidates:
            if admitted >= args.noun_limit:
                break
            if lemma in existing_expressions:
                _record_rejection(audit, "noun", lemma, "duplicate_expression")
                continue
            blocked_match = _match_blocked_lemma(lemma, vulgar_blocklist)
            if blocked_match is not None:
                _record_rejection(audit, "noun", lemma, "blocked_vulgar", **blocked_match)
                continue
            if lemma.endswith("ics"):
                _record_rejection(audit, "noun", lemma, "pluralia_tantum_like_surface")
                continue
            if _noun_expression_looks_plural(lemma):
                _record_rejection(audit, "noun", lemma, "plural_looking_singular_surface")
                continue
            if "_" in lemma or "-" in lemma or " " in lemma:
                _record_rejection(audit, "noun", lemma, "non_simple_surface")
                continue
            zipf_value = zipf_for_expression(lemma)
            if args.noun_zipf_min is not None and zipf_value < args.noun_zipf_min:
                _record_rejection(audit, "noun", lemma, "below_zipf_min", zipf=zipf_value)
                continue
            if args.noun_zipf_max is not None and zipf_value > args.noun_zipf_max:
                _record_rejection(audit, "noun", lemma, "above_zipf_max", zipf=zipf_value)
                continue
            quality_reason = _low_quality_noun_rejection(lemma)
            if quality_reason:
                _record_rejection(audit, "noun", lemma, quality_reason, zipf=zipf_value)
                continue
            bundle, bundle_confidence, subjects = _noun_bundle_for_lemma(lemma, args.lexicon)
            if bundle is None:
                _record_rejection(audit, "noun", lemma, "untyped_bundle", zipf=zipf_value, subjects=subjects)
                continue
            if bundle_confidence < args.noun_bundle_min_confidence:
                _record_rejection(
                    audit,
                    "noun",
                    lemma,
                    "low_bundle_confidence",
                    zipf=zipf_value,
                    bundle=bundle,
                    bundle_confidence=round(bundle_confidence, 3),
                    subjects=subjects,
                )
                continue
            entity_like, definitions = _noun_entity_like(lemma, args.lexicon)
            if entity_like:
                _record_rejection(
                    audit,
                    "noun",
                    lemma,
                    "entity_or_language_like",
                    zipf=zipf_value,
                    definitions=definitions[:3],
                )
                continue
            if bundle == "person":
                name_like, definitions = _noun_person_name_like(lemma, args.lexicon)
                if name_like:
                    _record_rejection(
                        audit,
                        "noun",
                        lemma,
                        "person_name_like",
                        zipf=zipf_value,
                        definitions=definitions[:3],
                    )
                    continue
            if bundle not in noun_templates:
                _record_rejection(audit, "noun", lemma, "missing_bundle_template", zipf=zipf_value, bundle=bundle)
                continue
            template_row = _choose_template(noun_templates[bundle])
            if template_row is None:
                _record_rejection(audit, "noun", lemma, "missing_singular_template", zipf=zipf_value, bundle=bundle)
                continue
            new_rows = _make_noun_rows(lemma, template_row, bundle)
            if not new_rows:
                _record_rejection(audit, "noun", lemma, "inflection_failed", zipf=zipf_value, bundle=bundle)
                continue
            if any(row["expression"] in existing_expressions for row in new_rows):
                _record_rejection(audit, "noun", lemma, "family_collision", zipf=zipf_value, bundle=bundle)
                continue
            overlay_rows.extend(new_rows)
            admitted += 1
            _record_admission(
                audit,
                "noun",
                lemma,
                zipf_value,
                template_row["expression"],
                bundle,
                len(new_rows),
            )
            for row in new_rows:
                signature = _row_signature_from_dict(row)
                manifest_rows.append({
                    "row_signature": signature,
                    "source": "overlay",
                    "source_lexicon": "oewn",
                    "source_lemma": lemma,
                    "inherited_template": template_row["expression"],
                    "validation_status": "validated",
                    "overlay_type": "noun",
                    "bundle": bundle,
                })
                existing_expressions.add(row["expression"])

    if args.include_verbs:
        template_index = _core_verb_template_index(_BASE_ROWS)
        sig_index = _sig_verb_template_index(template_index)
        special_template_index = _special_verb_template_index(_BASE_ROWS)
        intr_pp_template_index = _multiword_verb_template_index(_BASE_ROWS, "(S\\NP)/NP")
        intr_particle_template_index = _multiword_verb_template_index(_BASE_ROWS, "S\\NP")
        trans_particle_template_index = _multiword_verb_template_index(_BASE_ROWS, "(S\\NP)/NP")
        verb_entries = _load_verb_inventory(args.verb_inventory_path)
        rng.shuffle(verb_entries)
        admitted = 0
        for entry in verb_entries:
            if admitted >= args.verb_limit:
                break
            lemma = entry["lemma"]
            if lemma in existing_expressions:
                _record_rejection(audit, "verb", lemma, "duplicate_expression")
                continue
            blocked_match = _match_blocked_lemma(lemma, vulgar_blocklist)
            if blocked_match is not None:
                _record_rejection(audit, "verb", lemma, "blocked_vulgar", **blocked_match)
                continue
            if "_" in lemma or "-" in lemma or " " in lemma:
                _record_rejection(audit, "verb", lemma, "non_simple_surface")
                continue
            zipf_value = zipf_for_expression(lemma)
            if args.verb_zipf_min is not None and zipf_value < args.verb_zipf_min:
                _record_rejection(audit, "verb", lemma, "below_zipf_min", zipf=zipf_value)
                continue
            if args.verb_zipf_max is not None and zipf_value > args.verb_zipf_max:
                _record_rejection(audit, "verb", lemma, "above_zipf_max", zipf=zipf_value)
                continue
            quality_reason, quality_details = _head_ambiguous_intransitive_rejection(
                lemma,
                entry,
                zipf_value,
                args.lexicon,
            )
            if quality_reason:
                _record_rejection(audit, "verb", lemma, quality_reason, zipf=zipf_value, **quality_details)
                continue
            special_keys = _special_verb_candidate_keys(entry)
            core_template_bundles = []
            core_template_keys = _core_template_keys(entry["frame_types"])
            if core_template_keys:
                for template_key in core_template_keys:
                    core_template_bundles.append((template_key, None, template_index.get(template_key)))
            elif entry["frame_values"].get("intr_pp"):
                for prep in entry["frame_values"]["intr_pp"]:
                    if intr_pp_template_index.get(prep):
                        core_template_bundles.append(("intr_pp", prep, intr_pp_template_index[prep]))
                        break
            elif entry["frame_values"].get("intr_particle"):
                for particle in entry["frame_values"]["intr_particle"]:
                    if intr_particle_template_index.get(particle):
                        core_template_bundles.append(("intr_particle", particle, intr_particle_template_index[particle]))
                        break
            elif entry["frame_values"].get("trans_particle"):
                for particle in entry["frame_values"]["trans_particle"]:
                    if trans_particle_template_index.get(particle):
                        core_template_bundles.append(("trans_particle", particle, trans_particle_template_index[particle]))
                        break
            if not core_template_bundles and not special_keys:
                reason = "unsupported_frame_types"
                if entry["frame_values"].get("ditrans_pp"):
                    reason = "unsupported_ditrans_pp"
                _record_rejection(
                    audit,
                    "verb",
                    lemma,
                    reason,
                    zipf=zipf_value,
                    frame_types=sorted(entry["frame_types"]),
                    frame_values=entry["frame_values"],
                )
                continue
            admitted_any_sig = False
            saw_inflection_failure = False
            attempted_core_bundles = []
            missing_core_templates = []
            for template_key, suffix_token, families_for_key in core_template_bundles:
                attempted_core_bundles.append(template_key)
                if not families_for_key:
                    missing_core_templates.append(template_key)
                    continue
                sig_families = list(sig_index.get(template_key, {}).values()) if template_key in sig_index else []
                if not sig_families and families_for_key:
                    # Multiword families are indexed separately, so preserve the old fallback.
                    sig_families = [families_for_key[0]]
                if args.verb_templates_per_lemma > 0:
                    sig_families = sig_families[:args.verb_templates_per_lemma]
                for sig_family in sig_families:
                    template_label = sig_family[0]["root"]
                    new_rows = _make_verb_rows(lemma, sig_family, template_label, suffix_token=suffix_token)
                    new_rows = _apply_argument_structure_overrides(new_rows, entry)
                    new_rows = _filter_verb_rows_for_subject_quality(new_rows, lemma, args.lexicon)
                    if not new_rows:
                        saw_inflection_failure = True
                        continue
                    new_signatures = [_row_signature_from_dict(row) for row in new_rows]
                    if any(signature in existing_signatures for signature in new_signatures):
                        continue
                    overlay_rows.extend(new_rows)
                    admitted_any_sig = True
                    _record_admission(
                        audit,
                        "verb",
                        lemma,
                        zipf_value,
                        template_label,
                        template_key,
                        len(new_rows),
                    )
                    for row, signature in zip(new_rows, new_signatures):
                        manifest_rows.append({
                            "row_signature": signature,
                            "source": "overlay",
                            "source_lexicon": "verb_inventory",
                            "source_lemma": lemma,
                            "inherited_template": template_label,
                            "validation_status": "validated",
                            "overlay_type": "verb",
                            "bundle": template_key,
                        })
                        existing_signatures.add(signature)
            for special_key in special_keys:
                special_families = list(special_template_index.get(special_key, ()))
                if args.verb_templates_per_lemma > 0:
                    special_families = special_families[:args.verb_templates_per_lemma]
                for special_family, special_suffix in special_families:
                    template_label = special_family[0]["root"] or special_key
                    new_rows = _make_verb_rows(lemma, special_family, template_label, suffix_token=special_suffix)
                    new_rows = _apply_argument_structure_overrides(new_rows, entry)
                    new_rows = _filter_verb_rows_for_subject_quality(new_rows, lemma, args.lexicon)
                    if not new_rows:
                        saw_inflection_failure = True
                        continue
                    new_signatures = [_row_signature_from_dict(row) for row in new_rows]
                    if any(signature in existing_signatures for signature in new_signatures):
                        continue
                    overlay_rows.extend(new_rows)
                    admitted_any_sig = True
                    _record_admission(
                        audit,
                        "verb",
                        lemma,
                        zipf_value,
                        template_label,
                        special_key,
                        len(new_rows),
                    )
                    for row, signature in zip(new_rows, new_signatures):
                        manifest_rows.append({
                            "row_signature": signature,
                            "source": "overlay",
                            "source_lexicon": "verb_inventory",
                            "source_lemma": lemma,
                            "inherited_template": template_label,
                            "validation_status": "validated",
                            "overlay_type": "verb",
                            "bundle": special_key,
                        })
                        existing_signatures.add(signature)
            if not admitted_any_sig:
                if missing_core_templates and len(missing_core_templates) == len(core_template_bundles):
                    _record_rejection(
                        audit,
                        "verb",
                        lemma,
                        "missing_frame_template",
                        zipf=zipf_value,
                        bundle=",".join(missing_core_templates),
                        frame_values=entry["frame_values"],
                    )
                    continue
                if saw_inflection_failure:
                    _record_rejection(
                        audit,
                        "verb",
                        lemma,
                        "inflection_failed",
                        zipf=zipf_value,
                        bundle=",".join(attempted_core_bundles) if attempted_core_bundles else "special",
                    )
                else:
                    _record_rejection(
                        audit,
                        "verb",
                        lemma,
                        "family_collision",
                        zipf=zipf_value,
                        bundle=",".join(attempted_core_bundles) if attempted_core_bundles else "special",
                    )
            else:
                admitted += 1
            if admitted >= args.verb_limit:
                break

    if args.include_adjectives:
        adjective_templates = _adjective_template_index(_BASE_ROWS)
        special_adjective_templates = _special_adjective_template_index(_BASE_ROWS)
        adjective_candidates = _load_oe_adjective_lemmas(args.lexicon)
        rng.shuffle(adjective_candidates)
        admitted = 0
        for lemma in adjective_candidates:
            if admitted >= args.adjective_limit:
                break
            if lemma in existing_expressions:
                _record_rejection(audit, "adjective", lemma, "duplicate_expression")
                continue
            blocked_match = _match_blocked_lemma(lemma, vulgar_blocklist)
            if blocked_match is not None:
                _record_rejection(audit, "adjective", lemma, "blocked_vulgar", **blocked_match)
                continue
            if "_" in lemma or "-" in lemma or " " in lemma:
                _record_rejection(audit, "adjective", lemma, "non_simple_surface")
                continue
            zipf_value = zipf_for_expression(lemma)
            if args.adjective_zipf_min is not None and zipf_value < args.adjective_zipf_min:
                _record_rejection(audit, "adjective", lemma, "below_zipf_min", zipf=zipf_value)
                continue
            if args.adjective_zipf_max is not None and zipf_value > args.adjective_zipf_max:
                _record_rejection(audit, "adjective", lemma, "above_zipf_max", zipf=zipf_value)
                continue
            bundle, bundle_confidence, definitions = _adjective_bundle_for_lemma(lemma, args.lexicon)
            if bundle is None:
                _record_rejection(
                    audit,
                    "adjective",
                    lemma,
                    "unsupported_bundle",
                    zipf=zipf_value,
                    definitions=definitions[:3],
                )
                continue
            elif bundle_confidence < args.adjective_bundle_min_confidence:
                _record_rejection(
                    audit,
                    "adjective",
                    lemma,
                    "low_bundle_confidence",
                    zipf=zipf_value,
                    bundle=bundle,
                    bundle_confidence=round(bundle_confidence, 3),
                    definitions=definitions[:3],
                )
                continue
            template_rows = adjective_templates.get(bundle, ())
            template_row = _choose_template(template_rows)
            admitted_any = False
            if template_row is not None:
                new_rows = _make_adjective_rows(lemma, template_row)
                new_signatures = [_row_signature_from_dict(row) for row in new_rows]
                if new_rows and not any(signature in existing_signatures for signature in new_signatures):
                    overlay_rows.extend(new_rows)
                    admitted_any = True
                    _record_admission(
                        audit,
                        "adjective",
                        lemma,
                        zipf_value,
                        template_row["expression"],
                        bundle,
                        len(new_rows),
                    )
                    for row, signature in zip(new_rows, new_signatures):
                        manifest_rows.append({
                            "row_signature": signature,
                            "source": "overlay",
                            "source_lexicon": "oewn",
                            "source_lemma": lemma,
                            "inherited_template": template_row["expression"],
                            "validation_status": "validated",
                            "overlay_type": "adjective",
                            "bundle": bundle,
                        })
                        existing_expressions.add(row["expression"])
                        existing_signatures.add(signature)
            for special_key, special_rows in special_adjective_templates.items():
                special_template = _choose_template(special_rows)
                if special_template is None:
                    continue
                new_rows = _make_adjective_rows(lemma, special_template)
                new_signatures = [_row_signature_from_dict(row) for row in new_rows]
                if not new_rows or any(signature in existing_signatures for signature in new_signatures):
                    continue
                overlay_rows.extend(new_rows)
                admitted_any = True
                _record_admission(
                    audit,
                    "adjective",
                    lemma,
                    zipf_value,
                    special_template["expression"],
                    special_key,
                    len(new_rows),
                )
                for row, signature in zip(new_rows, new_signatures):
                    manifest_rows.append({
                        "row_signature": signature,
                        "source": "overlay",
                        "source_lexicon": "oewn",
                        "source_lemma": lemma,
                        "inherited_template": special_template["expression"],
                        "validation_status": "validated",
                        "overlay_type": "adjective",
                        "bundle": special_key,
                    })
                    existing_expressions.add(row["expression"])
                    existing_signatures.add(signature)
            if not admitted_any:
                _record_rejection(audit, "adjective", lemma, "missing_bundle_template", zipf=zipf_value, bundle=bundle)
                continue
            admitted += 1

    _write_rows(args.overlay_path, overlay_rows, _FIELDNAMES)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as handle:
        json.dump({"rows": manifest_rows}, handle, separators=(",", ":"))
    runtime_table = _base_runtime_table(_BASE_ROWS, overlay_rows)
    write_frequency_cache(build_frequency_cache(runtime_table), args.frequency_cache_path)
    if args.audit_path:
        audit_path = Path(args.audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_payload = {
            "config": {
                "lexicon": args.lexicon,
                "verb_inventory_path": args.verb_inventory_path,
                "vulgar_blocklist_path": args.vulgar_blocklist_path,
                "disable_vulgar_blocklist": args.disable_vulgar_blocklist,
                "seed": args.seed,
                "include_nouns": args.include_nouns,
                "include_verbs": args.include_verbs,
                "include_adjectives": args.include_adjectives,
                "noun_limit": args.noun_limit,
                "verb_limit": args.verb_limit,
                "adjective_limit": args.adjective_limit,
                "verb_templates_per_lemma": args.verb_templates_per_lemma,
                "noun_zipf_min": args.noun_zipf_min,
                "noun_zipf_max": args.noun_zipf_max,
                "noun_bundle_min_confidence": args.noun_bundle_min_confidence,
                "verb_zipf_min": args.verb_zipf_min,
                "verb_zipf_max": args.verb_zipf_max,
                "adjective_zipf_min": args.adjective_zipf_min,
                "adjective_zipf_max": args.adjective_zipf_max,
                "adjective_bundle_min_confidence": args.adjective_bundle_min_confidence,
            },
            "summary": _freeze_summary(audit),
            "admitted": audit["admitted"],
            "rejected": audit["rejected"],
        }
        with open(audit_path, "w") as handle:
            json.dump(audit_payload, handle, separators=(",", ":"))


def build_parser():
    parser = argparse.ArgumentParser(description="Build a BLiMP-compatible vocabulary overlay.")
    parser.add_argument("--overlay-path", default="vocabulary_overlay.csv")
    parser.add_argument("--manifest-path", default="vocabulary_overlay_manifest.json")
    parser.add_argument("--frequency-cache-path", default="outputs/cache/vocabulary_frequency_cache.json")
    parser.add_argument("--audit-path", default="outputs/cache/vocabulary_overlay_audit.json")
    parser.add_argument("--lexicon", default="oewn:2021")
    default_verb_inventory_path = str(DEFAULT_VERB_INVENTORY_PATH if DEFAULT_VERB_INVENTORY_PATH.exists() else LEGACY_VERB_INVENTORY_PATH)
    parser.add_argument("--verb-inventory-path", default=default_verb_inventory_path)
    parser.add_argument("--vulgar-blocklist-path", default=str(DEFAULT_VULGAR_BLOCKLIST_PATH))
    parser.add_argument("--disable-vulgar-blocklist", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-nouns", action="store_true")
    parser.add_argument("--include-verbs", action="store_true")
    parser.add_argument("--include-adjectives", action="store_true")
    parser.add_argument("--noun-limit", type=int, default=1000000)
    parser.add_argument("--verb-limit", type=int, default=1000000)
    parser.add_argument("--adjective-limit", type=int, default=1000000)
    parser.add_argument("--verb-templates-per-lemma", type=int, default=0)
    parser.add_argument("--noun-zipf-min", type=float, default=None)
    parser.add_argument("--noun-zipf-max", type=float, default=None)
    parser.add_argument("--noun-bundle-min-confidence", type=float, default=0.6)
    parser.add_argument("--verb-zipf-min", type=float, default=None)
    parser.add_argument("--verb-zipf-max", type=float, default=None)
    parser.add_argument("--adjective-zipf-min", type=float, default=None)
    parser.add_argument("--adjective-zipf-max", type=float, default=None)
    parser.add_argument("--adjective-bundle-min-confidence", type=float, default=0.6)
    return parser


_BASE_ROWS = _load_rows(BASE_VOCAB_PATH)
_FIELDNAMES = list(_BASE_ROWS[0].keys())


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.include_nouns and not args.include_verbs and not args.include_adjectives:
        args.include_nouns = True
        args.include_verbs = True
        args.include_adjectives = True
    build_overlay(args)


if __name__ == "__main__":
    main()
