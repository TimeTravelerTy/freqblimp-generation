from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from utils.vocab_sets import *

from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf
from generation_projects.blimp.overlay_guards import mismatching_nonpast_agreement_form
from generation_projects.blimp.overlay_guards import non_past_verb_rows


class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="morphology",
                         linguistics="subject_verb_agreement",
                         uid="distractor_agreement_relative_clause",
                         simple_lm_method=True,
                         one_prefix_method=True,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.all_reg_nouns = get_all_conjunctive([("category", "N"), ("irrpl", "")], all_common_nouns)
        self.safe_emb_verbs = all_transitive_verbs
        self.safe_mat_verbs = non_past_verb_rows()

    def sample(self):
        # The cat   that was      eating the mice is        sleeping
        #     Subj  Rel  Aux_Emb  V_emb  Obj_emb  Aux_agree V_mat_agree args
        # The cat   that was      eating the mice are           sleeping
        #     Subj  Rel  Aux_Emb  V_emb  Obj_emb  Aux_not_agree V_mat_not_agree args

        V_emb = None
        while V_emb is None:
            V_mat_agree = choice(filter_rows_for_active_zipf(self.safe_mat_verbs, "verb"))
            subj = N_to_DP_mutate(choice(filter_rows_for_active_zipf(get_matches_of(V_mat_agree, "arg_1", self.all_reg_nouns), "noun")))
            rel = choice(get_matched_by(subj, "arg_1", get_all("category_2", "rel")))
            V_mat_not_agree = mismatching_nonpast_agreement_form(V_mat_agree)

            if subj["pl"] == "1":
                obj_emb = N_to_DP_mutate(choice(filter_rows_for_active_zipf(get_matches_of(V_mat_not_agree, "arg_1", all_singular_nouns), "noun")))
            else:
                obj_emb = N_to_DP_mutate(choice(filter_rows_for_active_zipf(get_matches_of(V_mat_not_agree, "arg_1", all_plural_nouns), "noun")))

            try:
                V_emb = choice(filter_rows_for_active_zipf(get_matched_by(subj, "arg_1", get_matched_by(obj_emb, "arg_2", self.safe_emb_verbs)), "verb"))
            except IndexError:
                pass

        Aux_emb = return_aux(V_emb, subj)

        Auxs = require_aux_agree(V_mat_agree, subj)
        Aux_agree = Auxs["aux_agree"]
        Aux_not_agree = Auxs["aux_nonagree"]
        V_mat_args = verb_args_from_verb(V_mat_agree, subj=subj, aux=Aux_agree)

        if V_mat_agree["finite"] == "1":
            prefix = "%s %s %s %s %s" % (subj[0], rel[0], Aux_emb[0], V_emb[0], obj_emb[0])
            word_good = V_mat_agree[0]
            word_bad = V_mat_not_agree[0]
        else:
            prefix = "%s %s %s %s %s" % (subj[0], rel[0], Aux_emb[0], V_emb[0], obj_emb[0])
            word_good = Aux_agree
            word_bad = Aux_not_agree

        data = {
            "sentence_good": "%s %s %s %s %s %s %s %s." % (subj[0], rel[0], Aux_emb[0], V_emb[0], obj_emb[0], Aux_agree, V_mat_agree[0], join_args(V_mat_args["args"])),
            "sentence_bad": "%s %s %s %s %s %s %s %s." % (subj[0], rel[0], Aux_emb[0], V_emb[0], obj_emb[0], Aux_not_agree, V_mat_not_agree[0], join_args(V_mat_args["args"])),
            "one_prefix_prefix": prefix,
            "one_prefix_word_good": word_good,
            "one_prefix_word_bad": word_bad
        }
        return data, data["sentence_good"]


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
