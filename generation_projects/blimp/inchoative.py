from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice, uniform_choice

from generation_projects.blimp.overlay_guards import (
    causative_alternating_verb_rows,
    dp_buildable_nominal_rows,
    filter_rows_for_active_zipf,
    inchoative_bad_transitive_rows,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="inchoative",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)

        self.alternating_verbs = causative_alternating_verb_rows()
        self.non_alternating_transitives = inchoative_bad_transitive_rows()
        self.safe_nominals = dp_buildable_nominal_rows()
        self.all_singulars = get_all("sg", "1", self.safe_nominals)
        self.all_plurals = get_all("sg", "0", self.safe_nominals)
        zipf_bad_pool = filter_rows_for_active_zipf(self.non_alternating_transitives, "verb", fallback_on_empty=False)
        # Tail/xtail get too few globally Zipf-valid bad roots to maintain a
        # stable, diverse bad-side verb pool. Relax the bad slot when support is
        # tight instead of collapsing onto a few survivors.
        self.relax_bad_verb_zipf = len(np.unique(np.asarray(zipf_bad_pool["root"], dtype=str))) < 25
        self.max_sample_attempts = 512

    def sample(self):
        # The lamp has broken.
        # Subj     Aux V_inch
        # The lamp has pained.
        # Subj     Aux V_trans

        verb_pool = filter_rows_for_active_zipf(self.alternating_verbs, "verb", fallback_on_empty=False)
        if len(verb_pool) == 0:
            raise LexicalGapError("No xtail-compatible alternating verbs for inchoative")

        for _ in range(self.max_sample_attempts):
            V_inch = choice(verb_pool)
            if V_inch["category"] == "(S\\NP)/NP":
                if V_inch["3sg"] == "1":
                    subj_pool = filter_rows_for_active_zipf(
                        get_matches_of(V_inch, "arg_2", self.all_singulars), "noun", fallback_on_empty=False
                    )
                elif V_inch["pres"] == "1":
                    subj_pool = filter_rows_for_active_zipf(
                        get_matches_of(V_inch, "arg_2", self.all_plurals), "noun", fallback_on_empty=False
                    )
                else:
                    subj_pool = filter_rows_for_active_zipf(
                        get_matches_of(V_inch, "arg_2", self.safe_nominals), "noun", fallback_on_empty=False
                    )
            else:
                subj_pool = filter_rows_for_active_zipf(
                    get_matches_of(V_inch, "arg_1", self.safe_nominals), "noun", fallback_on_empty=False
                )
            if len(subj_pool) == 0:
                continue
            Subj = N_to_DP_mutate(choice(subj_pool))
            Aux = return_aux(V_inch, Subj)
            if Subj["sg"] == "1":
                safe_verbs = table_intersect1d(self.non_alternating_transitives, all_possibly_singular_verbs)
            else:
                safe_verbs = table_intersect1d(self.non_alternating_transitives, all_possibly_plural_verbs)
            bad_candidates = get_matched_by(Subj, "arg_2", get_matches_of(Aux, "arg_2", safe_verbs))
            if len(bad_candidates) == 0:
                continue
            if self.relax_bad_verb_zipf:
                V_trans = uniform_choice(bad_candidates)
            else:
                bad_candidates = filter_rows_for_active_zipf(
                    bad_candidates,
                    "verb",
                    fallback_on_empty=False,
                )
                if len(bad_candidates) == 0:
                    continue
                V_trans = choice(bad_candidates)
            break
        else:
            raise LexicalGapError("No xtail-compatible inchoative/transitive pair found after bounded retries")

        data = {
            "sentence_good": "%s %s %s." % (Subj[0], Aux[0], V_inch[0]),
            "sentence_bad": "%s %s %s." % (Subj[0], Aux[0], V_trans[0])
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
