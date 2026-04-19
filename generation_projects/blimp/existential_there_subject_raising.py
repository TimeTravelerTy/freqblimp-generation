from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from utils.exceptions import LexicalGapError
from functools import reduce

from generation_projects.blimp.overlay_guards import (
    control_subject_verb_rows,
    control_subject_adjective_rows,
    curated_template_expressions,
    existential_bad_control_subject_verb_rows,
    filter_plural_looking_singular_nouns,
    overlay_enabled,
    rows_matching_expressions,
    subject_raising_verb_rows,
    subject_raising_adjective_rows,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax_semantics",
                         linguistics="control_raising",
                         uid="existential_there_subject_raising",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)
        good_quantifiers_sg_str = ["a", "an", ""]
        good_quantifiers_pl_str = ["no", "some", "few", "fewer than three", "more than three", "many", "a lot of", ""]
        self.good_quantifiers_sg = reduce(np.union1d, [get_all("expression", s, all_determiners) for s in good_quantifiers_sg_str])
        self.good_quantifiers_pl = reduce(np.union1d, [get_all("expression", s, all_determiners) for s in good_quantifiers_pl_str])
        bad_emb_subjs = reduce(np.union1d, (all_relational_poss_nouns, all_proper_names, get_all("category", "NP")))
        self.safe_emb_subjs = filter_plural_looking_singular_nouns(np.setdiff1d(all_nominals, bad_emb_subjs))
        self.raising_verbs = subject_raising_verb_rows()
        curated_control_verbs = existential_bad_control_subject_verb_rows()
        if len(curated_control_verbs) > 0:
            self.control_verbs = curated_control_verbs
        else:
            self.control_verbs = np.setdiff1d(control_subject_verb_rows(), get_all("root", "fail_(S\\NP)/(S[to]\\N)"))
        self.raising_pred_rows = subject_raising_adjective_rows()
        self.control_pred_rows = control_subject_adjective_rows()
        self.raising_preds = tuple(curated_template_expressions("Adj_raising_subj"))
        self.control_preds = tuple(curated_template_expressions("Adj_control_subj"))

    def sample(self):
        # There does seem    to be a dog      eating an apple.
        # THERE aux  raising TO BE D emb_subj VP
        # There does try     to be a dog      eating an apple.
        # THERE aux  control TO BE D emb_subj VP

        emb_subj = N_to_DP_mutate(choice(self.safe_emb_subjs), determiner=False)
        D = choice(get_matched_by(emb_subj, "arg_1", self.good_quantifiers_sg)) \
            if emb_subj["sg"] == "1" \
            else choice(get_matched_by(emb_subj, "arg_1", self.good_quantifiers_pl))
        allow_negated = D[0] != "no" and D[0] != "some"

        if emb_subj["sg"] == "1":
            agree_verbs = all_possibly_singular_verbs
        else:
            agree_verbs = all_possibly_plural_verbs

        verbal_predicate = choice([True, False])
        if verbal_predicate:
            control_rows = rows_matching_expressions(self.control_verbs, agree_verbs)
            if len(control_rows) == 0:
                raise LexicalGapError("No compatible control-subject verbs for agreement class")
            control = choice(control_rows)
            aux = return_aux(control, emb_subj, allow_negated=allow_negated)
            control = control[0]
        else:
            control_row = choice(self.control_pred_rows) if overlay_enabled() and len(self.control_pred_rows) > 0 else None
            control = control_row[0] if control_row is not None else choice(self.control_preds)
            aux = return_copula(emb_subj, allow_negated=allow_negated)

        if verbal_predicate:
            raising_rows = rows_matching_expressions(self.raising_verbs, get_matches_of(aux, "arg_2", agree_verbs))
            if len(raising_rows) == 0:
                raise LexicalGapError("No compatible subject-raising verbs for auxiliary/agreement class")
            raising = choice(raising_rows)
            raising = raising[0]
        else:
            raising_row = choice(self.raising_pred_rows) if overlay_enabled() and len(self.raising_pred_rows) > 0 else None
            raising = raising_row[0] if raising_row is not None else choice(self.raising_preds)

        V = choice(get_matched_by(emb_subj, "arg_1", all_ing_verbs))
        args = verb_args_from_verb(V, subj=emb_subj, allow_negated=allow_negated)
        VP = V_to_VP_mutate(V, args=args, aux=False)

        data = {
            "sentence_good": "There %s %s to be %s %s %s." % (aux[0], raising, D[0], emb_subj[0], VP[0]),
            "sentence_bad": "There %s %s to be %s %s %s." % (aux[0], control, D[0], emb_subj[0], VP[0]),
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
