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
    confidence = count / sum(bundle_counts.values())
    return bundle, confidence, subjects


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


def _make_noun_rows(lemma, template_row):
    plural = _pluralize_noun(lemma)
    if template_row is None:
        return []
    if template_row.get("mass") == "1":
        singular_row = dict(template_row)
        return [_normalize_noun_row(singular_row, lemma)]
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
    return [singular_row, plural_row]


def _load_verb_inventory(path):
    payload = json.loads(Path(path).read_text())
    entries = []
    for entry in payload:
        lemma = str(entry.get("lemma", "")).strip().lower()
        if not lemma or not WORD_RE.fullmatch(lemma):
            continue
        frame_values = defaultdict(set)
        frame_types = {frame.get("type") for frame in entry.get("frames", []) if isinstance(frame, dict)}
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


def _core_template_key(frame_types):
    has_intr = "intr" in frame_types
    has_trans = "trans" in frame_types
    if has_intr and not has_trans:
        return "intr"
    if has_trans and not has_intr:
        return "trans"
    return None


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


def _template_suffix_token(template_family):
    bare_rows = [row for row in template_family if row["bare"] == "1"]
    template_row = bare_rows[0] if bare_rows else template_family[0]
    tokens = template_row["expression"].strip().split()
    if len(tokens) != 2:
        return None
    return tokens[-1].lower()


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
    root_suffix = "SNP" if category == "S\\NP" else "SNP_NP"
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


def _base_runtime_table(base_rows, overlay_rows):
    import numpy as np
    return np.array(
        [tuple(_row_value(row, field) for field, _dtype in data_type) for row in base_rows + overlay_rows],
        dtype=data_type,
    )


def build_overlay(args):
    rng = random.Random(args.seed)
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
            if lemma in VULGAR_BLOCKLIST:
                _record_rejection(audit, "noun", lemma, "blocked")
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
            if bundle not in noun_templates:
                _record_rejection(audit, "noun", lemma, "missing_bundle_template", zipf=zipf_value, bundle=bundle)
                continue
            template_row = _choose_template(noun_templates[bundle])
            if template_row is None:
                _record_rejection(audit, "noun", lemma, "missing_singular_template", zipf=zipf_value, bundle=bundle)
                continue
            new_rows = _make_noun_rows(lemma, template_row)
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
        intr_pp_template_index = _multiword_verb_template_index(_BASE_ROWS, "(S\\NP)/NP")
        intr_particle_template_index = _multiword_verb_template_index(_BASE_ROWS, "S\\NP")
        trans_particle_template_index = _multiword_verb_template_index(_BASE_ROWS, "(S\\NP)/NP")
        verb_entries = _load_verb_inventory(args.verb_inventory_path)
        rng.shuffle(verb_entries)
        admitted = 0
        template_offsets = defaultdict(int)
        for entry in verb_entries:
            if admitted >= args.verb_limit:
                break
            lemma = entry["lemma"]
            if lemma in existing_expressions:
                _record_rejection(audit, "verb", lemma, "duplicate_expression")
                continue
            if lemma in VULGAR_BLOCKLIST:
                _record_rejection(audit, "verb", lemma, "blocked")
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
            suffix_token = None
            template_key = _core_template_key(entry["frame_types"])
            families_for_key = None
            if template_key is not None:
                families_for_key = template_index.get(template_key)
            elif entry["frame_values"].get("intr_pp"):
                for prep in entry["frame_values"]["intr_pp"]:
                    if intr_pp_template_index.get(prep):
                        template_key = "intr_pp"
                        suffix_token = prep
                        families_for_key = intr_pp_template_index[prep]
                        break
            elif entry["frame_values"].get("intr_particle"):
                for particle in entry["frame_values"]["intr_particle"]:
                    if intr_particle_template_index.get(particle):
                        template_key = "intr_particle"
                        suffix_token = particle
                        families_for_key = intr_particle_template_index[particle]
                        break
            elif entry["frame_values"].get("trans_particle"):
                for particle in entry["frame_values"]["trans_particle"]:
                    if trans_particle_template_index.get(particle):
                        template_key = "trans_particle"
                        suffix_token = particle
                        families_for_key = trans_particle_template_index[particle]
                        break
            if template_key is None:
                reason = "ambiguous_core_frames" if "intr" in entry["frame_types"] and "trans" in entry["frame_types"] else "unsupported_frame_types"
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
            if not families_for_key:
                _record_rejection(
                    audit,
                    "verb",
                    lemma,
                    "missing_frame_template",
                    zipf=zipf_value,
                    bundle=template_key,
                    frame_values=entry["frame_values"],
                )
                continue
            template_count = min(args.verb_templates_per_lemma, len(families_for_key))
            admitted_family = False
            for template_rank in range(template_count):
                template_offset = (template_offsets[template_key] + template_rank) % len(families_for_key)
                template_family = families_for_key[template_offset]
                template_label = template_family[0]["root"]
                new_rows = _make_verb_rows(lemma, template_family, template_label, suffix_token=suffix_token)
                if not new_rows:
                    if not admitted_family:
                        _record_rejection(audit, "verb", lemma, "inflection_failed", zipf=zipf_value, bundle=template_key)
                    continue
                new_signatures = [_row_signature_from_dict(row) for row in new_rows]
                if any(signature in existing_signatures for signature in new_signatures):
                    continue
                overlay_rows.extend(new_rows)
                admitted += 1
                admitted_family = True
                template_offsets[template_key] += 1
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
                if admitted >= args.verb_limit:
                    break
            if not admitted_family:
                _record_rejection(audit, "verb", lemma, "family_collision", zipf=zipf_value, bundle=template_key)
            if admitted >= args.verb_limit:
                break

    _write_rows(args.overlay_path, overlay_rows, _FIELDNAMES)
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as handle:
        json.dump({"rows": manifest_rows}, handle, indent=2, sort_keys=True)
    runtime_table = _base_runtime_table(_BASE_ROWS, overlay_rows)
    write_frequency_cache(build_frequency_cache(runtime_table), args.frequency_cache_path)
    if args.audit_path:
        audit_path = Path(args.audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_payload = {
            "config": {
                "lexicon": args.lexicon,
                "verb_inventory_path": args.verb_inventory_path,
                "seed": args.seed,
                "include_nouns": args.include_nouns,
                "include_verbs": args.include_verbs,
                "noun_limit": args.noun_limit,
                "verb_limit": args.verb_limit,
                "verb_templates_per_lemma": args.verb_templates_per_lemma,
                "noun_zipf_min": args.noun_zipf_min,
                "noun_zipf_max": args.noun_zipf_max,
                "noun_bundle_min_confidence": args.noun_bundle_min_confidence,
                "verb_zipf_min": args.verb_zipf_min,
                "verb_zipf_max": args.verb_zipf_max,
            },
            "summary": _freeze_summary(audit),
            "admitted": audit["admitted"],
            "rejected": audit["rejected"],
        }
        with open(audit_path, "w") as handle:
            json.dump(audit_payload, handle, indent=2, sort_keys=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Build a BLiMP-compatible vocabulary overlay.")
    parser.add_argument("--overlay-path", default="vocabulary_overlay.csv")
    parser.add_argument("--manifest-path", default="vocabulary_overlay_manifest.json")
    parser.add_argument("--frequency-cache-path", default="outputs/cache/vocabulary_frequency_cache.json")
    parser.add_argument("--audit-path", default="outputs/cache/vocabulary_overlay_audit.json")
    parser.add_argument("--lexicon", default="oewn:2021")
    parser.add_argument("--verb-inventory-path", default="../blimp-rare/data/processed/verb_inventory_pruned_particles.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--include-nouns", action="store_true")
    parser.add_argument("--include-verbs", action="store_true")
    parser.add_argument("--noun-limit", type=int, default=50)
    parser.add_argument("--verb-limit", type=int, default=50)
    parser.add_argument("--verb-templates-per-lemma", type=int, default=1)
    parser.add_argument("--noun-zipf-min", type=float, default=None)
    parser.add_argument("--noun-zipf-max", type=float, default=3.4)
    parser.add_argument("--noun-bundle-min-confidence", type=float, default=0.6)
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
