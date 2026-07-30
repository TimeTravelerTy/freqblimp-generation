from collections import Counter

from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.frequency import zipf_for_expression

from generation_projects.blimp.overlay_guards import (
    choose_matching_row,
    choose_row_for_active_zipf_by_source_lemma,
    dp_buildable_nominal_rows,
    drop_argument_bad_verb_rows,
    drop_argument_good_verb_rows,
    filter_rows_by_zipf_distance,
    filter_rows_for_active_zipf,
    source_lemma_for_row,
)

class CSCGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="drop_argument",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)

        self.drop_arg_transitive = drop_argument_good_verb_rows()
        self.drop_arg_bad_transitive = drop_argument_bad_verb_rows()
        self.safe_subjects = dp_buildable_nominal_rows()
        self.max_sample_attempts = 512
        # Keep the lexical contrast from carrying a systematic Zipf advantage.
        # This operates after the ordinary per-regime filtering, so it never
        # admits an out-of-regime verb merely to complete a pair.
        self.max_pair_zipf_gap = 0.15
        self.good_lemma_counts = Counter()
        self.bad_lemma_counts = Counter()
        active_good_rows = filter_rows_for_active_zipf(
            self.drop_arg_transitive,
            "verb",
            fallback_on_empty=False,
        )
        self.good_lemma_universe = frozenset(
            source_lemma_for_row(row).lower() for row in active_good_rows
        )
        active_bad_rows = filter_rows_for_active_zipf(
            self.drop_arg_bad_transitive,
            "verb",
            fallback_on_empty=False,
        )
        self.bad_lemma_universe = frozenset(
            source_lemma_for_row(row).lower() for row in active_bad_rows
        )
        # Compare a small reservoir of compatible pairs before accepting an
        # item. This improves control-side balance without creating a hard cap
        # that can deadlock on argument/frequency compatibility.
        self.pair_candidate_trials = 8

    def _surface_sentence(self, subj, aux, verb):
        text = remove_extra_whitespace(f"{subj[0]} {aux[0]} {verb[0]}")
        return text[0].upper() + text[1:] + "."

    def sample(self):
        # The bear has attacked.
        # Subj     Aux V_non_strict
        # The bear has injured.
        # Subj     Aux V_strict

        best_candidate = None
        best_score = None
        valid_candidate_count = 0
        for _ in range(self.max_sample_attempts):
            try:
                V_non_strict = choose_row_for_active_zipf_by_source_lemma(
                    self.drop_arg_transitive,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No xtail-compatible drop-argument verbs",
                    minimum_candidates=64,
                    lemma_counts=self.good_lemma_counts,
                    lemma_count_slack=1,
                )
                good_lemma = source_lemma_for_row(V_non_strict).lower()
                Subj = N_to_DP_mutate(choose_matching_row(
                    V_non_strict,
                    "arg_1",
                    self.safe_subjects,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible drop-argument subject",
                    minimum_candidates=10,
                ))
                Aux = return_aux(V_non_strict, Subj)
                strict_candidates = get_matched_by(
                    Subj,
                    "arg_1",
                    get_matches_of(Aux, "arg_2", self.drop_arg_bad_transitive),
                )
                strict_candidates = filter_rows_for_active_zipf(
                    strict_candidates,
                    "verb",
                    fallback_on_empty=False,
                )
                strict_candidates = filter_rows_by_zipf_distance(
                    strict_candidates,
                    zipf_for_expression(V_non_strict["expression"]),
                    self.max_pair_zipf_gap,
                )
                V_strict = choose_row_for_active_zipf_by_source_lemma(
                    strict_candidates,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No frequency-matched strict transitive contrast verb",
                    lemma_counts=self.bad_lemma_counts,
                )
                bad_lemma = source_lemma_for_row(V_strict).lower()
                score = (
                    self.good_lemma_counts.get(good_lemma, 0),
                    self.bad_lemma_counts.get(bad_lemma, 0),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_candidate = (Subj, Aux, V_non_strict, V_strict)
                valid_candidate_count += 1
                minimum_good_count = min(
                    self.good_lemma_counts.get(lemma, 0)
                    for lemma in self.good_lemma_universe
                )
                minimum_bad_count = min(
                    self.bad_lemma_counts.get(lemma, 0)
                    for lemma in self.bad_lemma_universe
                )
                if score == (minimum_good_count, minimum_bad_count):
                    break
                if valid_candidate_count >= self.pair_candidate_trials:
                    break
            except LexicalGapError:
                continue
        if best_candidate is None:
            raise LexicalGapError("No xtail-compatible drop_argument pair found after bounded retries")
        Subj, Aux, V_non_strict, V_strict = best_candidate

        data = {
            "sentence_good": self._surface_sentence(Subj, Aux, V_non_strict),
            "sentence_bad": self._surface_sentence(Subj, Aux, V_strict),
        }
        self.good_lemma_counts[source_lemma_for_row(V_non_strict).lower()] += 1
        self.bad_lemma_counts[source_lemma_for_row(V_strict).lower()] += 1
        return data, data["sentence_good"]


def build_generator():
    return CSCGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
