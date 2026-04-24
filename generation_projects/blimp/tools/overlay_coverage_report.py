import argparse
import csv
import os

from generation_projects.blimp.build_overlay import (
    _load_oe_adjective_lemmas,
    _load_oe_lemmas,
    _load_verb_inventory,
)
from utils.frequency import zipf_for_expression
from utils.randomize import SamplingPolicy, configure_sampling_policy, clear_sampling_policy


def _xtail_source_lemmas(lexicon, xt_min, xt_max):
    noun_lemmas = {
        lemma for lemma in _load_oe_lemmas("n", lexicon)
        if xt_min <= zipf_for_expression(lemma) <= xt_max
    }
    adjective_lemmas = {
        lemma for lemma in _load_oe_adjective_lemmas(lexicon)
        if xt_min <= zipf_for_expression(lemma) <= xt_max
    }
    verb_lemmas = {
        entry["lemma"] for entry in _load_verb_inventory("generation_projects/blimp/verb_inventory.json")
        if xt_min <= zipf_for_expression(entry["lemma"]) <= xt_max
    }
    return noun_lemmas, verb_lemmas, adjective_lemmas


def _xtail_overlay_lemmas(overlay_path, xt_min, xt_max):
    noun_lemmas = set()
    verb_lemmas = set()
    adjective_lemmas = set()
    with open(overlay_path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("noun") == "1":
                lemma = row.get("singularform") or row["expression"]
                if xt_min <= zipf_for_expression(lemma) <= xt_max:
                    noun_lemmas.add(lemma)
            if row.get("verb") == "1":
                root = row.get("root", "")
                lemma = root.split("_overlay_", 1)[0] if "_overlay_" in root else row["expression"]
                if xt_min <= zipf_for_expression(lemma) <= xt_max:
                    verb_lemmas.add(lemma)
            if row.get("category") == "N/N" and row.get("category_2") in {"Adj", "adjective"}:
                lemma = row["expression"]
                if xt_min <= zipf_for_expression(lemma) <= xt_max:
                    adjective_lemmas.add(lemma)
    return noun_lemmas, verb_lemmas, adjective_lemmas


def _configure_overlay_runtime(args):
    os.environ["FREQBLIMP_USE_OVERLAY"] = "1"
    os.environ["FREQBLIMP_VOCAB_OVERLAY"] = args.overlay_path
    os.environ["FREQBLIMP_OVERLAY_MANIFEST"] = args.overlay_manifest_path
    os.environ["FREQBLIMP_FREQUENCY_CACHE"] = args.frequency_cache_path
    policy = SamplingPolicy(
        seed=0,
        controlled_pos=("noun", "verb", "adjective"),
        zipf_min={"noun": args.xtail_min, "verb": args.xtail_min, "adjective": args.xtail_min},
        zipf_max={"noun": args.xtail_max, "verb": args.xtail_max, "adjective": args.xtail_max},
        overlay_enabled=True,
        overlay_path=args.overlay_path,
        overlay_manifest_path=args.overlay_manifest_path,
        frequency_cache_path=args.frequency_cache_path,
    )
    configure_sampling_policy(policy)


def _probe_argument_structure():
    from generation_projects.blimp.causative import build_generator as build_causative
    from generation_projects.blimp.inchoative import build_generator as build_inchoative
    from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf
    from utils.vocab_sets import all_nominals
    from utils.vocab_table import get_matches_of

    results = {}
    for uid, build in (("causative", build_causative), ("inchoative", build_inchoative)):
        generator = build()
        rows = filter_rows_for_active_zipf(generator.alternating_verbs, "verb", fallback_on_empty=False)
        feasible = []
        blocked = []
        for row in rows:
            key = "arg_1"
            if uid == "inchoative" and row["category"] == "(S\\NP)/NP":
                key = "arg_2"
            matches = filter_rows_for_active_zipf(
                get_matches_of(row, key, all_nominals),
                "noun",
                fallback_on_empty=False,
            )
            record = {
                "expression": str(row["expression"]),
                "category": str(row["category"]),
                "matches": len(matches),
                "sample": [str(x["expression"]) for x in matches[:6]],
            }
            if len(matches) > 0:
                feasible.append(record)
            else:
                blocked.append(record)
        results[uid] = {"feasible": feasible, "blocked": blocked}
    return results


def main():
    parser = argparse.ArgumentParser(description="Report xtail source-vs-overlay coverage and argument-structure feasibility.")
    parser.add_argument("--overlay-path", default="vocabulary_overlay.csv")
    parser.add_argument("--overlay-manifest-path", default="vocabulary_overlay_manifest.json")
    parser.add_argument("--frequency-cache-path", default="outputs/cache/vocabulary_frequency_cache.json")
    parser.add_argument("--lexicon", default="oewn:2021")
    parser.add_argument("--xtail-min", type=float, default=1.2)
    parser.add_argument("--xtail-max", type=float, default=2.2)
    args = parser.parse_args()

    source_nouns, source_verbs, source_adjs = _xtail_source_lemmas(args.lexicon, args.xtail_min, args.xtail_max)
    overlay_nouns, overlay_verbs, overlay_adjs = _xtail_overlay_lemmas(args.overlay_path, args.xtail_min, args.xtail_max)

    print("xtail_source noun=%d verb=%d adjective=%d" % (len(source_nouns), len(source_verbs), len(source_adjs)))
    print("xtail_overlay noun=%d verb=%d adjective=%d" % (len(overlay_nouns), len(overlay_verbs), len(overlay_adjs)))
    print("xtail_missing noun=%d verb=%d adjective=%d" % (
        len(source_nouns - overlay_nouns),
        len(source_verbs - overlay_verbs),
        len(source_adjs - overlay_adjs),
    ))

    _configure_overlay_runtime(args)
    try:
        probes = _probe_argument_structure()
    finally:
        clear_sampling_policy()

    for uid, info in probes.items():
        print("%s feasible=%d blocked=%d" % (uid, len(info["feasible"]), len(info["blocked"])))
        for item in info["blocked"][:10]:
            print("  blocked", item["expression"], item["category"], item["matches"])
        for item in info["feasible"][:5]:
            print("  feasible", item["expression"], item["category"], item["matches"], item["sample"])


if __name__ == "__main__":
    main()
