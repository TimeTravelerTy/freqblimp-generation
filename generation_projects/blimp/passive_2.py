from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from generation_projects.blimp.overlay_guards import (
    choose_matching_row,
    choose_matched_by_row,
    choose_row_for_active_zipf,
    nonpassivizable_participle_rows,
    passivizable_participle_rows,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="passive_2",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.intransitive = nonpassivizable_participle_rows()
        self.transitive = passivizable_participle_rows()
        self.max_sample_attempts = 512

    def sample(self):
        # The girl was attacked.
        # NP1      be  V_trans
        # The girl was smiled.
        # NP1      be  V_intrans

        for _ in range(self.max_sample_attempts):
            try:
                V_intrans = choose_row_for_active_zipf(
                    self.intransitive,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No xtail-compatible purely-intransitive verb",
                    minimum_candidates=10,
                )
                NP1 = N_to_DP_mutate(choose_matching_row(
                    V_intrans,
                    "arg_1",
                    all_nominals,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible passive patient",
                ))
                V_trans = choose_matched_by_row(
                    NP1,
                    "arg_2",
                    self.transitive,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No regime-compatible passive verb",
                    minimum_candidates=10,
                )
                be = return_copula(NP1)
            except LexicalGapError:
                continue
            break
        else:
            raise LexicalGapError("No xtail-compatible passive pair found after bounded retries")

        data = {
            "sentence_good": "%s %s %s." % (NP1[0], be[0], V_trans[0]),
            "sentence_bad": "%s %s %s." % (NP1[0], be[0], V_intrans[0])
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
