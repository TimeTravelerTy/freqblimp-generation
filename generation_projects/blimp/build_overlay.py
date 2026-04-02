import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import wn
from lemminflect import getInflection

from utils.frequency import zipf_for_expression
from utils.vocab_table import BASE_VOCAB_PATH, build_frequency_cache, write_frequency_cache
from utils.data_type import data_type


WORD_RE = re.compile(r"^[a-z]+$")
VULGAR_BLOCKLIST = {
    "asshole",
    "bitch",
    "cock",
    "cunt",
    "dick",
    "dildo",
    "fag",
    "faggot",
    "fuck",
    "motherfucker",
    "penis",
    "pussy",
    "slut",
    "twat",
    "vagina",
    "whore",
}
NOUN_SUBJECT_TO_BUNDLE = {
    "noun.person": "person",
    "noun.artifact": "artifact",
    "noun.communication": "document",
    "noun.cognition": "conceptual",
    "noun.food": "food_count",
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
}
FIELD_ALIASES = {
    "synonym/hypernym/hyponym": "synonym_hypernym_hyponym",
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


def _noun_subject(lemma, lexicon):
    try:
        synsets = wn.synsets(lemma, pos="n", lexicon=lexicon)
    except Exception:
        return None
    if not synsets:
        return None
    metadata = synsets[0].metadata() if callable(getattr(synsets[0], "metadata", None)) else {}
    subject = metadata.get("subject") if isinstance(metadata, dict) else None
    if subject == "noun.artifact":
        definition = (synsets[0].definition() or "").lower()
        if "vehicle" in definition:
            return "noun.vehicle"
    return subject


def _noun_bundle_for_lemma(lemma, lexicon):
    subject = _noun_subject(lemma, lexicon)
    if subject == "noun.vehicle":
        return "vehicle"
    return NOUN_SUBJECT_TO_BUNDLE.get(subject)


def _noun_template_index(rows):
    index = defaultdict(list)
    for row in rows:
        if row["category"] != "N" or row["properNoun"] != "0":
            continue
        for bundle, fields in NOUN_BUNDLE_FIELDS.items():
            if all(row.get(key, "") == value for key, value in fields.items()):
                index[bundle].append(row)
    return index


def _choose_template(rows):
    singular = [row for row in rows if row["sg"] == "1"]
    singular.sort(key=lambda row: row["expression"])
    if not singular:
        return None
    return singular[0]


def _pluralize_noun(lemma):
    forms = getInflection(lemma, tag="NNS")
    return forms[0].lower() if forms else None


def _make_noun_rows(lemma, template_row):
    plural = _pluralize_noun(lemma)
    if template_row is None:
        return []
    if template_row.get("mass") == "1":
        singular_row = dict(template_row)
        singular_row["expression"] = lemma
        singular_row["frequent"] = "1"
        singular_row["pluralform"] = ""
        singular_row["singularform"] = ""
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
    singular_row["expression"] = lemma
    singular_row["pluralform"] = plural
    singular_row["singularform"] = ""
    singular_row["frequent"] = "1"
    plural_row["expression"] = plural
    plural_row["singularform"] = lemma
    plural_row["pluralform"] = ""
    plural_row["frequent"] = "1"
    return [singular_row, plural_row]


def _load_verb_inventory(path):
    payload = json.loads(Path(path).read_text())
    entries = []
    for entry in payload:
        lemma = str(entry.get("lemma", "")).strip().lower()
        if not lemma or not WORD_RE.fullmatch(lemma):
            continue
        frame_types = {frame.get("type") for frame in entry.get("frames", []) if isinstance(frame, dict)}
        entries.append({"lemma": lemma, "frame_types": frame_types})
    return entries


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


def _make_verb_rows(lemma, template_family):
    category = template_family[0]["category"]
    root_suffix = "SNP" if category == "S\\NP" else "SNP_NP"
    new_root = "%s_overlay_%s" % (lemma, root_suffix)
    rows = []
    for template_row in template_family:
        form = _verb_form_for_row(lemma, template_row)
        if not form:
            return []
        row = dict(template_row)
        row["expression"] = form
        row["root"] = new_root
        row["frequent"] = "1"
        rows.append(row)
    return rows


def _base_runtime_table(base_rows, overlay_rows):
    import numpy as np
    return np.array(
        [tuple(_row_value(row, field) for field, _dtype in data_type) for row in base_rows + overlay_rows],
        dtype=data_type,
    )


def build_overlay(args):
    rng = random.Random(args.seed)
    existing_expressions = {row["expression"] for row in _BASE_ROWS}
    overlay_rows = []
    manifest_rows = []

    if args.include_nouns:
        noun_templates = _noun_template_index(_BASE_ROWS)
        noun_candidates = _load_oe_lemmas("n", args.lexicon)
        rng.shuffle(noun_candidates)
        admitted = 0
        for lemma in noun_candidates:
            if admitted >= args.noun_limit:
                break
            if lemma in existing_expressions or lemma in VULGAR_BLOCKLIST:
                continue
            if "_" in lemma or "-" in lemma or " " in lemma:
                continue
            zipf_value = zipf_for_expression(lemma)
            if args.noun_zipf_min is not None and zipf_value < args.noun_zipf_min:
                continue
            if args.noun_zipf_max is not None and zipf_value > args.noun_zipf_max:
                continue
            bundle = _noun_bundle_for_lemma(lemma, args.lexicon)
            if bundle not in noun_templates:
                continue
            template_row = _choose_template(noun_templates[bundle])
            new_rows = _make_noun_rows(lemma, template_row)
            if not new_rows:
                continue
            if any(row["expression"] in existing_expressions for row in new_rows):
                continue
            overlay_rows.extend(new_rows)
            admitted += 1
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
        verb_entries = _load_verb_inventory(args.verb_inventory_path)
        rng.shuffle(verb_entries)
        admitted = 0
        for entry in verb_entries:
            if admitted >= args.verb_limit:
                break
            lemma = entry["lemma"]
            if lemma in existing_expressions or lemma in VULGAR_BLOCKLIST:
                continue
            if "_" in lemma or "-" in lemma or " " in lemma:
                continue
            zipf_value = zipf_for_expression(lemma)
            if args.verb_zipf_min is not None and zipf_value < args.verb_zipf_min:
                continue
            if args.verb_zipf_max is not None and zipf_value > args.verb_zipf_max:
                continue
            template_key = None
            if "trans" in entry["frame_types"]:
                template_key = "trans"
            elif "intr" in entry["frame_types"]:
                template_key = "intr"
            if template_key is None or not template_index.get(template_key):
                continue
            template_family = template_index[template_key][admitted % len(template_index[template_key])]
            new_rows = _make_verb_rows(lemma, template_family)
            if not new_rows:
                continue
            if any(row["expression"] in existing_expressions for row in new_rows):
                continue
            overlay_rows.extend(new_rows)
            admitted += 1
            for row in new_rows:
                signature = _row_signature_from_dict(row)
                manifest_rows.append({
                    "row_signature": signature,
                    "source": "overlay",
                    "source_lexicon": "verb_inventory",
                    "source_lemma": lemma,
                    "inherited_template": template_family[0]["root"],
                    "validation_status": "validated",
                    "overlay_type": "verb",
                    "bundle": template_key,
                })
                existing_expressions.add(row["expression"])

    _write_rows(args.overlay_path, overlay_rows, _FIELDNAMES)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as handle:
        json.dump({"rows": manifest_rows}, handle, indent=2, sort_keys=True)
    runtime_table = _base_runtime_table(_BASE_ROWS, overlay_rows)
    write_frequency_cache(build_frequency_cache(runtime_table), args.frequency_cache_path)


def build_parser():
    parser = argparse.ArgumentParser(description="Build a BLiMP-compatible vocabulary overlay.")
    parser.add_argument("--overlay-path", default="vocabulary_overlay.csv")
    parser.add_argument("--manifest-path", default="vocabulary_overlay_manifest.json")
    parser.add_argument("--frequency-cache-path", default="outputs/cache/vocabulary_frequency_cache.json")
    parser.add_argument("--lexicon", default="oewn:2021")
    parser.add_argument("--verb-inventory-path", default="../blimp-rare/data/processed/verb_inventory_pruned_particles.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-nouns", action="store_true")
    parser.add_argument("--include-verbs", action="store_true")
    parser.add_argument("--noun-limit", type=int, default=50)
    parser.add_argument("--verb-limit", type=int, default=50)
    parser.add_argument("--noun-zipf-min", type=float, default=None)
    parser.add_argument("--noun-zipf-max", type=float, default=3.4)
    parser.add_argument("--verb-zipf-min", type=float, default=None)
    parser.add_argument("--verb-zipf-max", type=float, default=3.4)
    return parser


_BASE_ROWS = _load_rows(BASE_VOCAB_PATH)
_FIELDNAMES = list(_BASE_ROWS[0].keys())


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.include_nouns and not args.include_verbs:
        args.include_nouns = True
        args.include_verbs = True
    build_overlay(args)


if __name__ == "__main__":
    main()
