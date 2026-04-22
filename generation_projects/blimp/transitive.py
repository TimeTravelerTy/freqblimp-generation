from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="transitive",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)

        self.strict_intransitive = get_all("category", "S\\NP", get_all("strict_intrans", "1"))
        transitive_exprs = set(map(str, all_transitive_verbs["expression"]))
        keep_mask = ~np.isin(np.asarray(self.strict_intransitive["expression"], dtype=str), list(transitive_exprs))
        self.strict_intransitive = self.strict_intransitive[keep_mask]
        self.max_sample_attempts = 512

    def sample(self):
        # The bear has attacked the girl.
        # Subj     Aux V_trans  obj
        # The bear has smiled    the girl.
        # Subj     Aux V_intrans obj

        verb_pool = filter_rows_for_active_zipf(all_transitive_verbs, "verb", fallback_on_empty=False)
        if len(verb_pool) == 0:
            raise LexicalGapError("No regime-compatible transitive verbs")

        for _ in range(self.max_sample_attempts):
            V_trans = choice(verb_pool)
            subj_pool = filter_rows_for_active_zipf(
                get_matches_of(V_trans, "arg_1", all_nominals), "noun", fallback_on_empty=False
            )
            if len(subj_pool) == 0:
                continue
            Subj = N_to_DP_mutate(choice(subj_pool))
            Aux = return_aux(V_trans, Subj)
            obj_pool = filter_rows_for_active_zipf(
                get_matches_of(V_trans, "arg_2", all_nominals), "noun", fallback_on_empty=False
            )
            if len(obj_pool) == 0:
                continue
            Obj = N_to_DP_mutate(choice(obj_pool))
            bad_pool = filter_rows_for_active_zipf(
                get_matched_by(Subj, "arg_1", get_matches_of(Aux, "arg_2", self.strict_intransitive)),
                "verb",
                fallback_on_empty=False,
            )
            if len(bad_pool) == 0:
                continue
            V_intrans = choice(bad_pool)
            break
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
