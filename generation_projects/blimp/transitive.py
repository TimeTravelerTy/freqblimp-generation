from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import (
    choose_matching_row,
    choose_matched_by_row,
    choose_row_for_active_zipf,
    filter_plural_looking_singular_nouns,
    pure_strict_intransitive_rows,
    pure_transitive_rows,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="transitive",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)

        self.strict_intransitive = pure_strict_intransitive_rows()
        self.transitive_verbs = pure_transitive_rows()
        self.safe_nominals = filter_plural_looking_singular_nouns(all_nominals)
        self.max_sample_attempts = 512

    def sample(self):
        # The bear has attacked the girl.
        # Subj     Aux V_trans  obj
        # The bear has smiled    the girl.
        # Subj     Aux V_intrans obj

        for _ in range(self.max_sample_attempts):
            try:
                V_trans = choose_row_for_active_zipf(
                    self.transitive_verbs,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No regime-compatible transitive verbs",
                )
                Subj = N_to_DP_mutate(choose_matching_row(
                    V_trans,
                    "arg_1",
                    self.safe_nominals,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible transitive subject",
                ))
                Aux = return_aux(V_trans, Subj)
                Obj = N_to_DP_mutate(choose_matching_row(
                    V_trans,
                    "arg_2",
                    self.safe_nominals,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible transitive object",
                ))
                V_intrans = choose_matched_by_row(
                    Subj,
                    "arg_1",
                    get_matches_of(Aux, "arg_2", self.strict_intransitive),
                    "verb",
                    fallback_on_empty=False,
                    error_message="No regime-compatible intransitive contrast verb",
                )
                break
            except LexicalGapError:
                continue
        else:
            raise LexicalGapError("No regime-compatible transitive pair found after bounded retries")

        data = {
            "sentence_good": "%s %s %s %s." % (Subj[0], Aux[0], V_trans[0], Obj[0]),
            "sentence_bad": "%s %s %s %s." % (Subj[0], Aux[0], V_intrans[0], Obj[0]),
            "two_prefix_prefix_good": "%s %s %s" % (Subj[0], Aux[0], V_trans[0]),
            "two_prefix_prefix_bad": "%s %s %s" % (Subj[0], Aux[0], V_intrans[0]),
            "two_prefix_word": Obj[0].strip().split(" ")[0]
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
