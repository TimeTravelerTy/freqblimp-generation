import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generation_projects.blimp import build_overlay as build_overlay_module
from generation_projects.blimp.overlay_coverage import (
    DONE_OVERLAY_CASES,
    DONE_PARADIGMS,
    REMAINING_CONTENT_WORD_TARGETS,
    RUNTIME_VERIFIED_WITHOUT_NEW_HOOKS,
    SKIP_FOR_OVERLAY,
)
from utils import data_generator
from utils.randomize import SamplingPolicy, clear_sampling_policy, configure_sampling_policy, set_trace_recording_enabled
from utils.vocab_table import clear_query_caches


class BlimpGeneratorSmokeTests(unittest.TestCase):
    def _build_generator(self, stem):
        module = importlib.import_module("generation_projects.blimp.%s" % stem)
        if hasattr(module, "build_generator"):
            return module.build_generator()
        generator_classes = []
        for _name, value in inspect.getmembers(module, inspect.isclass):
            if value in {
                data_generator.Generator,
                data_generator.BenchmarkGenerator,
                data_generator.ScalarImplicatureGenerator,
                data_generator.PresuppositionGenerator,
            }:
                continue
            if issubclass(value, data_generator.Generator):
                generator_classes.append(value)
        self.assertTrue(generator_classes, stem)
        generator_classes.sort(key=lambda cls: cls.__name__)
        return generator_classes[0]()

    def test_all_blimp_generators_construct(self):
        root = Path("generation_projects/blimp")
        ignored = {"__init__", "run", "build_overlay", "sbatch_generator", "prune_flagged_verb_frames", "overlay_guards", "overlay_coverage"}
        failures = []
        for path in sorted(root.glob("*.py")):
            stem = path.stem
            if stem in ignored:
                continue
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
        root = Path("generation_projects/blimp")
        ignored = {"__init__", "run", "build_overlay", "sbatch_generator", "prune_flagged_verb_frames", "overlay_guards", "overlay_coverage"}
        helper = BlimpGeneratorSmokeTests()
        return sorted(
            helper._build_generator(path.stem).uid
            for path in root.glob("*.py")
            if path.stem not in ignored
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


if __name__ == "__main__":
    unittest.main()
