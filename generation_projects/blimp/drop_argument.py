from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import (
    dp_buildable_nominal_rows,
    drop_argument_bad_verb_rows,
    drop_argument_good_verb_rows,
    filter_rows_for_active_zipf,
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

    def _surface_sentence(self, subj, aux, verb):
        text = remove_extra_whitespace(f"{subj[0]} {aux[0]} {verb[0]}")
        return text[0].upper() + text[1:] + "."

    def sample(self):
        # The bear has attacked.
        # Subj     Aux V_non_strict
        # The bear has injured.
        # Subj     Aux V_strict

        verb_pool = filter_rows_for_active_zipf(self.drop_arg_transitive, "verb", fallback_on_empty=False)
        if len(verb_pool) == 0:
            raise LexicalGapError("No xtail-compatible drop-argument verbs")

        for _ in range(self.max_sample_attempts):
            V_non_strict = choice(verb_pool)
            subj_pool = filter_rows_for_active_zipf(
                get_matches_of(V_non_strict, "arg_1", self.safe_subjects), "noun", fallback_on_empty=False
            )
            if len(subj_pool) == 0:
                continue
            Subj = N_to_DP_mutate(choice(subj_pool))
            Aux = return_aux(V_non_strict, Subj)
            V_strict_pool = filter_rows_for_active_zipf(
                get_matched_by(Subj, "arg_1", get_matches_of(Aux, "arg_2", self.drop_arg_bad_transitive)),
                "verb",
                fallback_on_empty=False,
            )
            if len(V_strict_pool) == 0:
                continue
            V_strict = choice(V_strict_pool)
            break
        else:
            raise LexicalGapError("No xtail-compatible drop_argument pair found after bounded retries")

        data = {
            "sentence_good": self._surface_sentence(Subj, Aux, V_non_strict),
            "sentence_bad": self._surface_sentence(Subj, Aux, V_strict),
        }
        return data, data["sentence_good"]


def build_generator():
    return CSCGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
