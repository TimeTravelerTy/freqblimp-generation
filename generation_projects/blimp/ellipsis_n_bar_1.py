from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from utils.vocab_sets import *
from generation_projects.blimp.overlay_guards import (
    safe_transitive_verb_rows,
    simple_common_noun_rows,
    strict_zipf_rows,
)

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="ellipsis",
                         uid="ellipsis_n_bar_1",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.safe_verbs = safe_transitive_verb_rows()
        self.plural_dets = [("two", "three"),
                            ("three", "two"),
                            ("four", "five"),
                            ("five", "four"),
                            ("several", "a few"),
                            ("a few", "several"),
                            ("many", "few"),
                            ("few", "many"),
                            ("a few", "a lot"),
                            ("a lot of", "a few"),
                            ("no", "some"),
                            ("three", "more"),
                            ("three", "fewer"),
                            ("three", "at least as many"),
                            ("three", "almost as many"),
                            ]

        self.singular_dets = [("one", "three"),
                            ("one", "two"),
                            ("one", "five"),
                            ("one", "four"),
                            ("one", "a few"),
                            ("one", "several"),
                            ("one", "few"),
                            ("one", "many"),
                            ("one", "a lot"),
                            ("one", "some"),
                            ("one", "more"),
                            ("one", "at least as many")
                            ]
        self.safe_objs = simple_common_noun_rows()

    def sample(self):
        # John  has  had two green cups and Jane  has had three.
        # Subj1 Aux1 V   D1  Adj   Obj  AND Subj2 Aux2 V   D2
        # John  has  had two cups and Jane  has  had three green.
        # Subj1 Aux1 V   D1  Obj  AND Subj2 Aux2 V   D2    Adj
        V = choice(strict_zipf_rows(self.safe_verbs, "verb", minimum_candidates=10))
        subj_pool = strict_zipf_rows(get_matches_of(V, "arg_1", all_nominals), "noun", minimum_candidates=10)
        Subj1 = choice(subj_pool)
        Subj2 = choice(subj_pool, avoid=Subj1)
        Subj1 = N_to_DP_mutate(Subj1)
        Subj2 = N_to_DP_mutate(Subj2)
        Aux1 = return_aux(V, subj=Subj1)
        if is_match_disj(Subj2, Aux1["arg_1"]):
            Aux2 = Aux1
        else:
            Aux2 = return_aux(V, subj=Subj2)
        Obj = choice(strict_zipf_rows(get_matches_of(V, "arg_2", get_all("mass", "0", self.safe_objs)), "noun", minimum_candidates=10))
        Adj = choice(strict_zipf_rows(get_matched_by(Obj, "arg_1", all_adjectives), "adjective", minimum_candidates=10))
        if Obj["pl"] == "0":
            (D1, D2) = random.choice(self.singular_dets)
        else:
            (D1, D2) = random.choice(self.plural_dets)

        data = {
            "sentence_good": "%s %s %s %s %s %s and %s %s %s %s." % (Subj1[0], Aux1[0], V[0], D1, Adj[0], Obj[0], Subj2[0], Aux2[0], V[0], D2),
            "sentence_bad": "%s %s %s %s %s and %s %s %s %s %s." % (Subj1[0], Aux1[0], V[0], D1, Obj[0], Subj2[0], Aux2[0], V[0], D2, Adj[0]),
        }
        return data, data["sentence_good"]


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
