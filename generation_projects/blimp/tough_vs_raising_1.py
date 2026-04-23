from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *

from generation_projects.blimp.overlay_guards import (
    choose_matching_row,
    choose_row_for_active_zipf,
    dp_buildable_nominal_rows,
    subject_raising_adjective_rows,
    tough_adjective_rows,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax_semantics",
                         linguistics="control_raising",
                         uid="tough_vs_raising_1",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)

        self.raising_preds = subject_raising_adjective_rows()
        self.tough_preds = tough_adjective_rows()
        self.strict_transitive = get_all("strict_trans", "1")
        self.safe_nominals = dp_buildable_nominal_rows()
        self.max_sample_attempts = 128

    def sample(self):
        # The hamburger is tough    to devour
        # Subj          be A_tough  TO V
        # The hamburger is likely   to devour
        # Subj          be A_raise  TO V

        for _ in range(self.max_sample_attempts):
            try:
                A_tough = choose_row_for_active_zipf(
                    self.tough_preds,
                    "adjective",
                    fallback_on_empty=False,
                    error_message="No regime-compatible tough adjectives",
                    minimum_candidates=10,
                )
                A_raising = choose_row_for_active_zipf(
                    self.raising_preds,
                    "adjective",
                    fallback_on_empty=False,
                    error_message="No regime-compatible raising adjectives",
                    minimum_candidates=10,
                )
                V = choose_matching_row(
                    A_tough,
                    "arg_1",
                    self.strict_transitive,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No regime-compatible tough_vs_raising_1 verb",
                    minimum_candidates=10,
                )
                subj = N_to_DP_mutate(choose_matching_row(
                    V,
                    "arg_2",
                    self.safe_nominals,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible tough_vs_raising_1 subject",
                    minimum_candidates=10,
                ))
                be = return_copula(subj)
                break
            except LexicalGapError:
                continue
        else:
            raise LexicalGapError("No regime-compatible tough_vs_raising_1 sample found after bounded retries")

        data = {
            "sentence_good": "%s %s %s to %s." % (subj[0], be[0], A_tough[0], V[0]),
            "sentence_bad": "%s %s %s to %s." % (subj[0], be[0], A_raising[0], V[0]),
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
