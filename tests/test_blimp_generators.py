import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from generation_projects.blimp import build_overlay as build_overlay_module
from generation_projects.blimp.overlay_coverage import (
    DONE_OVERLAY_CASES,
    DONE_PARADIGMS,
    REMAINING_CONTENT_WORD_TARGETS,
    RUNTIME_VERIFIED_WITHOUT_NEW_HOOKS,
    SKIP_FOR_OVERLAY,
)
from generation_projects.blimp.registry import build_generator_from_stem, generator_stems
from utils import data_generator
from utils.exceptions import LexicalGapError
from utils.randomize import SamplingPolicy, clear_sampling_policy, configure_sampling_policy, set_trace_recording_enabled
from utils.vocab_table import clear_query_caches


class BlimpGeneratorSmokeTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("FREQBLIMP_MAX_FAILURES", None)

    def _build_generator(self, stem):
        return build_generator_from_stem(stem)

    def test_all_blimp_generators_construct(self):
        failures = []
        for stem in generator_stems():
            try:
                self._build_generator(stem)
            except Exception as exc:  # pragma: no cover - surfaced via assertion below
                failures.append("%s: %r" % (stem, exc))
        self.assertEqual(failures, [])

    def test_previously_broken_generators_generate_one_pair(self):
        clear_sampling_policy()
        for stem in (
            "existential_there_subject_raising",
            "existential_there_quantifiers_1",
            "ellipsis_n_bar_2",
            "superlative_quantifiers_1",
        ):
            generator = self._build_generator(stem)
            with self.subTest(generator=stem):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = str(Path(tmpdir) / ("%s.jsonl" % stem))
                    generator.generate_paradigm(number_to_generate=1, absolute_path=output_path)
                    self.assertTrue(Path(output_path).exists(), stem)

    def test_generate_paradigm_allows_many_recoverable_failures(self):
        class EventuallySuccessfulGenerator(data_generator.Generator):
            def __init__(self):
                super().__init__()
                self.uid = "eventually_successful"
                self.data_fields = ["sentence_good", "sentence_bad"]
                self.attempts = 0

            def make_metadata_dict(self):
                return {"UID": self.uid}

            def sample(self):
                self.attempts += 1
                if self.attempts <= 11:
                    raise LexicalGapError("try again")
                return (
                    {
                        "sentence_good": "The teacher smiled.",
                        "sentence_bad": "The teacher smile.",
                    },
                    "The teacher smiled.",
                )

        clear_sampling_policy()
        generator = EventuallySuccessfulGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "eventually_successful.jsonl"
            manifest_path = Path(tmpdir) / "eventually_successful.manifest.json"
            generator.generate_paradigm(number_to_generate=1, absolute_path=str(output_path))
            payload = json.loads(output_path.read_text().splitlines()[0])
            manifest = json.loads(manifest_path.read_text())
        self.assertEqual(payload["sentence_good"], "The teacher smiled.")
        self.assertEqual(manifest["run"]["generated_pairs"], 1)
        self.assertEqual(manifest["run"]["failures"], 11)
        self.assertIsNone(manifest["run"]["failure_limit"])


class BlimpFrequencyGuardTests(unittest.TestCase):
    def tearDown(self):
        clear_sampling_policy()
        clear_query_caches()

    def test_curated_control_raising_families_respect_head_band(self):
        from generation_projects.blimp.overlay_guards import (
            subject_raising_adjective_rows,
            tough_adjective_rows,
        )

        configure_sampling_policy(
            SamplingPolicy(
                seed=0,
                controlled_pos=("adjective",),
                zipf_min={"adjective": 4.0},
                zipf_max={"adjective": 7.0},
            )
        )
        clear_query_caches()

        tough_rows = tough_adjective_rows()
        raising_rows = subject_raising_adjective_rows()

        self.assertGreaterEqual(len(tough_rows), 10)
        self.assertGreaterEqual(len(raising_rows), 10)
        self.assertNotIn("illuminating", set(map(str, tough_rows["expression"])))
        self.assertNotIn("reputed", set(map(str, raising_rows["expression"])))

    def test_bare_nominal_dps_synthesize_agreement_features(self):
        from utils.constituent_building import N_to_DP_mutate
        from utils.vocab_sets import all_auxs, all_ing_verbs, all_nominals
        from utils.vocab_table import get_all, get_matched_by

        subj = N_to_DP_mutate(get_all("expression", "computer", all_nominals)[0], determiner=False)
        subj_features = set(str(subj["arg_1"]).split("^"))
        self.assertIn("sg=1", subj_features)
        self.assertIn("mass=0", subj_features)
        self.assertIn("start_with_vowel=0", subj_features)
        self.assertIn("properNoun=0", subj_features)

        verb = get_all("expression", "failing", all_ing_verbs)[0]
        aux_matches = get_matched_by(verb, "arg_2", get_matched_by(subj, "arg_1", all_auxs))
        aux_expressions = set(map(str, aux_matches["expression"]))
        self.assertNotIn("are", aux_expressions)
        self.assertNotIn("were", aux_expressions)
        self.assertTrue(aux_expressions & {"is", "was", "isn't", "wasn't"})

    def test_indexed_table_zipf_cache_keys_do_not_alias_reused_index_buffers(self):
        from generation_projects.blimp.overlay_guards import (
            filter_rows_for_active_zipf,
            subject_raising_verb_rows,
        )
        from utils.vocab_table import IndexedTable

        configure_sampling_policy(
            SamplingPolicy(
                seed=0,
                controlled_pos=("verb",),
                zipf_min={"verb": 4.0},
                zipf_max={"verb": 7.0},
            )
        )
        clear_query_caches()

        rows = subject_raising_verb_rows()
        ing_positions = np.flatnonzero(np.asarray(rows["ing"], dtype=str) == "1")
        pres_positions = np.flatnonzero(
            (np.asarray(rows["finite"], dtype=str) == "1")
            & (np.asarray(rows["pres"], dtype=str) == "1")
        )[:len(ing_positions)]
        self.assertGreater(len(ing_positions), 0)

        shared_indices = np.array(ing_positions, dtype=np.int64)
        ing_view = IndexedTable(rows, shared_indices)
        ing_filtered = filter_rows_for_active_zipf(ing_view, "verb", fallback_on_empty=False)
        ing_expressions = tuple(map(str, ing_filtered["expression"]))

        shared_indices[:] = pres_positions
        pres_view = IndexedTable(rows, shared_indices)
        pres_filtered = filter_rows_for_active_zipf(pres_view, "verb", fallback_on_empty=False)
        pres_expressions = tuple(map(str, pres_filtered["expression"]))

        self.assertEqual(set(ing_expressions), set(map(str, rows[ing_positions]["expression"])))
        self.assertEqual(set(pres_expressions), set(map(str, rows[pres_positions]["expression"])))
        self.assertNotEqual(set(ing_expressions), set(pres_expressions))

    def test_global_verb_pool_excludes_blocked_overlay_artifacts(self):
        from utils.vocab_sets import all_verbs

        self.assertNotIn("according", set(map(str, all_verbs["expression"])))


class BlimpOverlaySmokeTests(unittest.TestCase):
    _STEM_BY_UID = {
        "animate_subject_trans": "animate_subject_transitive",
    }

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        root = Path(cls._tmpdir.name)
        cls.overlay_path = str(root / "overlay.csv")
        cls.manifest_path = str(root / "manifest.json")
        cls.cache_path = str(root / "frequency_cache.json")
        cls.audit_path = str(root / "audit.json")
        args = build_overlay_module.build_parser().parse_args(
            [
                "--overlay-path", cls.overlay_path,
                "--manifest-path", cls.manifest_path,
                "--frequency-cache-path", cls.cache_path,
                "--audit-path", cls.audit_path,
                "--include-nouns",
                "--include-verbs",
                "--include-adjectives",
                "--noun-limit", "60",
                "--verb-limit", "80",
                "--adjective-limit", "30",
                "--verb-templates-per-lemma", "1",
            ]
        )
        build_overlay_module.build_overlay(args)
        clear_query_caches()

    @classmethod
    def tearDownClass(cls):
        clear_sampling_policy()
        clear_query_caches()
        cls._tmpdir.cleanup()

    def tearDown(self):
        clear_sampling_policy()
        clear_query_caches()

    def _configure_overlay_policy(self, controlled_pos, zipf_min=1.0, zipf_max=3.2):
        os.environ["FREQBLIMP_USE_OVERLAY"] = "1"
        os.environ["FREQBLIMP_VOCAB_OVERLAY"] = self.overlay_path
        os.environ["FREQBLIMP_OVERLAY_MANIFEST"] = self.manifest_path
        os.environ["FREQBLIMP_FREQUENCY_CACHE"] = self.cache_path
        set_trace_recording_enabled(True)
        configure_sampling_policy(
            SamplingPolicy(
                seed=0,
                controlled_pos=controlled_pos,
                zipf_min={pos: zipf_min for pos in controlled_pos},
                zipf_max={pos: zipf_max for pos in controlled_pos},
                overlay_enabled=True,
                overlay_path=self.overlay_path,
                overlay_manifest_path=self.manifest_path,
                frequency_cache_path=self.cache_path,
            )
        )
        clear_query_caches()

    def _generate_one_pair_with_overlay(self, stem, controlled_pos, zipf_min=1.0, zipf_max=3.2):
        self._configure_overlay_policy(controlled_pos, zipf_min=zipf_min, zipf_max=zipf_max)
        generator = BlimpGeneratorSmokeTests()._build_generator(self._STEM_BY_UID.get(stem, stem))
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / ("%s.jsonl" % stem)
            generator.generate_paradigm(number_to_generate=1, absolute_path=str(output_path))
            payload = json.loads(output_path.read_text().splitlines()[0])
        return payload

    def _generate_pairs_via_cli_with_overlay(self, stem, controlled_pos, number_to_generate=3):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / ("%s.jsonl" % stem)
            cmd = [
                sys.executable,
                "-m",
                "generation_projects.blimp.run",
                stem,
                "--number-to-generate",
                str(number_to_generate),
                "--output-path",
                str(output_path),
                "--use-overlay",
                "--overlay-path",
                self.overlay_path,
                "--overlay-manifest-path",
                self.manifest_path,
                "--frequency-cache-path",
                self.cache_path,
                "--zipf-min-all",
                "1.0",
                "--zipf-max-all",
                "3.2",
                "--controlled-pos",
                *controlled_pos,
                "--no-progress",
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]

    def _all_blimp_uids(self):
        helper = BlimpGeneratorSmokeTests()
        return sorted(
            helper._build_generator(stem).uid
            for stem in generator_stems()
        )

    def test_overlay_manifest_contains_special_bundles(self):
        with open(self.manifest_path) as handle:
            manifest = json.load(handle)
        bundles = {row.get("bundle") for row in manifest.get("rows", [])}
        for bundle in (
            "Adj_clausal",
            "Adj_control_subj",
            "Adj_raising_subj",
            "Adj_tough",
            "V_control_object",
            "V_control_subj",
            "V_raising_object",
            "V_raising_subj",
        ):
            with self.subTest(bundle=bundle):
                self.assertIn(bundle, bundles)

    def test_frequency_cache_is_compact_expression_registry(self):
        with open(self.cache_path) as handle:
            payload = json.load(handle)
        self.assertEqual(payload.get("format_version"), 2)
        self.assertIn("expressions", payload)
        self.assertNotIn("rows", payload)
        self.assertTrue(payload["expressions"])

    def test_overlay_targeted_paradigms_generate(self):
        for stem, controlled_pos in DONE_OVERLAY_CASES:
            with self.subTest(generator=stem):
                payload = self._generate_one_pair_with_overlay(
                    stem,
                    controlled_pos=controlled_pos,
                    zipf_min=1.0,
                    zipf_max=3.2,
                )
                self.assertIn("sentence_good", payload)
                self.assertIn("sentence_bad", payload)
                self.assertNotEqual(payload["sentence_good"], payload["sentence_bad"])

    def test_overlay_coverage_buckets_partition_known_uids(self):
        covered = set(DONE_PARADIGMS) | set(REMAINING_CONTENT_WORD_TARGETS) | set(SKIP_FOR_OVERLAY)
        self.assertEqual(set(self._all_blimp_uids()), covered)
        self.assertEqual(set(), set(DONE_PARADIGMS) & set(REMAINING_CONTENT_WORD_TARGETS))
        self.assertEqual(set(), set(DONE_PARADIGMS) & set(SKIP_FOR_OVERLAY))
        self.assertEqual(set(), set(REMAINING_CONTENT_WORD_TARGETS) & set(SKIP_FOR_OVERLAY))

    def test_overlay_coverage_has_no_remaining_content_word_targets(self):
        self.assertEqual(REMAINING_CONTENT_WORD_TARGETS, ())

    def test_runtime_verified_no_hook_cases_expand_with_overlay(self):
        controlled_pos_by_uid = {
            "ellipsis_n_bar_1": ("noun", "adjective"),
            "transitive": ("verb",),
            "intransitive": ("verb",),
        }
        for uid in RUNTIME_VERIFIED_WITHOUT_NEW_HOOKS:
            with self.subTest(generator=uid):
                payloads = self._generate_pairs_via_cli_with_overlay(
                    uid,
                    controlled_pos=controlled_pos_by_uid[uid],
                    number_to_generate=3,
                )
                self.assertTrue(payloads)
                self.assertTrue(
                    any(
                        any(choice.get("source") == "overlay" for choice in payload.get("meta", {}).get("good_choices", ()))
                        or any(choice.get("source") == "overlay" for choice in payload.get("meta", {}).get("bad_choices", ()))
                        for payload in payloads
                    )
                )
                for payload in payloads:
                    self.assertNotEqual(payload["sentence_good"], payload["sentence_bad"])

    def test_runtime_vocab_switches_to_overlay_in_process(self):
        import utils.vocab_table as vocab_table

        os.environ.pop("FREQBLIMP_USE_OVERLAY", None)
        os.environ.pop("FREQBLIMP_VOCAB_OVERLAY", None)
        clear_query_caches()
        base_vocab = vocab_table.get_runtime_vocab()
        self.assertEqual(type(base_vocab).__name__, "memmap")

        os.environ["FREQBLIMP_USE_OVERLAY"] = "1"
        os.environ["FREQBLIMP_VOCAB_OVERLAY"] = self.overlay_path
        clear_query_caches()
        overlay_vocab = vocab_table.get_runtime_vocab()
        self.assertEqual(type(overlay_vocab).__name__, "ConcatTable")
        self.assertEqual(len(overlay_vocab.parts), 2)


class VocabTableMemorySafetyTests(unittest.TestCase):
    def tearDown(self):
        for name in (
            "FREQBLIMP_MAX_FREQUENCY_CACHE_LOAD_BYTES",
            "FREQBLIMP_MAX_MANIFEST_LOAD_BYTES",
            "FREQBLIMP_MAX_JSON_LOAD_BYTES",
            "FREQBLIMP_ALLOW_LARGE_JSON_LOADS",
        ):
            os.environ.pop(name, None)
        clear_query_caches()

    def test_large_frequency_cache_load_is_skipped_by_default(self):
        import utils.vocab_table as vocab_table

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"format_version": 2, "expressions": {"hello": 4.2}}, handle)
            path = handle.name
        try:
            os.environ["FREQBLIMP_MAX_FREQUENCY_CACHE_LOAD_BYTES"] = "1"
            self.assertEqual(vocab_table._load_frequency_cache(path), {})
        finally:
            os.remove(path)

    def test_large_overlay_manifest_load_is_skipped_by_default(self):
        import utils.vocab_table as vocab_table

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"rows": [{"row_signature": "x", "source": "overlay"}]}, handle)
            path = handle.name
        try:
            os.environ["FREQBLIMP_MAX_MANIFEST_LOAD_BYTES"] = "1"
            self.assertEqual(vocab_table._load_overlay_manifest(path), {})
        finally:
            os.remove(path)

    def test_composite_query_merges_base_and_overlay_without_building_runtime_table(self):
        import utils.vocab_table as vocab_table

        sample_source = vocab_table.get_runtime_vocab()
        if isinstance(sample_source, tuple):
            sample_source = sample_source[0]
        sample_source = getattr(sample_source, "source", sample_source)
        base_rows = sample_source[:2].copy()
        overlay_rows = sample_source[:2].copy()
        base_rows["expression"][0] = "base_only_token"
        base_rows["expression"][1] = "shared_token"
        overlay_rows["expression"][0] = "overlay_only_token"
        overlay_rows["expression"][1] = "shared_token"

        combined = (base_rows, overlay_rows)
        merged = vocab_table.get_all("expression", "shared_token", combined)
        self.assertEqual(len(merged), 2)
        overlay_only = vocab_table.get_all("expression", "overlay_only_token", combined)
        self.assertEqual(len(overlay_only), 1)

    def test_large_expression_lookup_avoids_building_full_label_index(self):
        import utils.vocab_table as vocab_table

        table = np.zeros(100001, dtype=[("expression", "U16"), ("root", "U16")])
        table["expression"][:-1] = "filler"
        table["expression"][-1] = "needle"
        table["root"][:] = "root"

        with mock.patch.object(vocab_table, "_get_label_index", side_effect=AssertionError("unexpected full index build")):
            result = vocab_table.get_all("expression", "needle", table)

        self.assertEqual(len(result), 1)
        self.assertEqual(str(result[0]["expression"]), "needle")

if __name__ == "__main__":
    unittest.main()
