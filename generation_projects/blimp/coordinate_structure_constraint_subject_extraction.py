from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from utils.vocab_sets import *

from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf

class CSCGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="island_effects",
                         uid="coordinate_structure_constraint_subject_extraction",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=True)

    def sample(self):
        # What and bananas did  you eat?
        # wh   and N2      V_do N1  V1

        # What did  you eat and bananas?
        # wh   V_do N1  V1  and N2

        V1 = choice(filter_rows_for_active_zipf(all_non_finite_transitive_verbs, "verb"))
        N1 = N_to_DP_mutate(choice(filter_rows_for_active_zipf(get_matches_of(V1, "arg_1", all_nouns), "noun")))
        V_do = return_aux(V1, N1, allow_negated=False)
        N2 = N_to_DP_mutate(choice(filter_rows_for_active_zipf(get_matches_of(V1, "arg_2", all_nouns), "noun")))
        wh = choice(get_matched_by(N2, "arg_1", all_wh_words))

        data = {
            "sentence_good": "%s and %s %s %s %s?" % (wh[0], N2[0], V_do[0], N1[0], V1[0]),
            "sentence_bad": "%s %s %s %s and %s?" % (wh[0], V_do[0], N1[0], V1[0], N2[0])
        }
        return data, data["sentence_good"]


def build_generator():
    return CSCGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
