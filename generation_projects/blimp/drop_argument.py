from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import (
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

    def sample(self):
        # The bear has attacked.
        # Subj     Aux V_non_strict
        # The bear has injured.
        # Subj     Aux V_strict

        V_non_strict = choice(filter_rows_for_active_zipf(self.drop_arg_transitive, "verb"))
        Subj = N_to_DP_mutate(choice(get_matches_of(V_non_strict, "arg_1", all_nominals)))
        Aux = return_aux(V_non_strict, Subj)
        V_strict = choice(filter_rows_for_active_zipf(
            get_matched_by(Subj, "arg_1", get_matches_of(Aux, "arg_2", self.drop_arg_bad_transitive)),
            "verb",
        ))

        data = {
            "sentence_good": "%s %s %s." % (Subj[0], Aux[0], V_non_strict[0]),
            "sentence_bad": "%s %s %s." % (Subj[0], Aux[0], V_strict[0])
        }
        return data, data["sentence_good"]


def build_generator():
    return CSCGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
