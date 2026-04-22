from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import (
    subject_raising_adjective_rows,
    tough_adjective_rows,
    tough_vs_raising_2_outer_verb_rows,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax_semantics",
                         linguistics="control_raising",
                         uid="tough_vs_raising_2",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)

        self.raising_preds = subject_raising_adjective_rows()
        self.tough_preds = np.setdiff1d(tough_adjective_rows(), get_all("expression", "ready"))
        self.safe_verbs = tough_vs_raising_2_outer_verb_rows()
        self.max_sample_attempts = 128

    def sample(self):
        # The hamburger is likely     to taste good
        # Subj          be A_raising  TO VP
        # The hamburger is tough    to taste good
        # Subj          be A_tough  TO VP

        for _ in range(self.max_sample_attempts):
            A_tough = choice(self.tough_preds)
            A_raising = choice(self.raising_preds)
            V = choice(self.safe_verbs)
            VP = V_to_VP_mutate(V, aux=False)
            vp_toks = VP[0].split()
            if len(vp_toks) >= 3 and vp_toks[0] == vp_toks[2] and vp_toks[1] == "to":
                continue
            subj = N_to_DP_mutate(choice(get_matches_of(V, "arg_1")))
            be = return_copula(subj)
            break
        else:
            raise LexicalGapError("No non-degenerate tough_vs_raising_2 sample found after bounded retries")

        data = {
            "sentence_good": "%s %s %s to %s." % (subj[0], be[0], A_raising[0], VP[0]),
            "sentence_bad": "%s %s %s to %s." % (subj[0], be[0], A_tough[0], VP[0]),
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
