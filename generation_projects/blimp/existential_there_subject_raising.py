from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from utils.exceptions import LexicalGapError
from functools import reduce

from generation_projects.blimp.overlay_guards import (
    choose_row_for_active_zipf,
    control_subject_verb_rows,
    control_subject_adjective_rows,
    curated_template_expressions,
    existential_bad_control_subject_verb_rows,
    filter_plural_looking_singular_nouns,
    filter_rows_for_active_zipf,
    overlay_enabled,
    rows_matching_inflection,
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
        self.max_sample_attempts = 512

    def _surface_sentence(self, aux, predicate, determiner, emb_subj, vp):
        text = remove_extra_whitespace(f"There {aux[0]} {predicate} to be {determiner[0]} {emb_subj[0]} {vp[0]}")
        return text + "."

    def sample(self):
        # There does seem    to be a dog      eating an apple.
        # THERE aux  raising TO BE D emb_subj VP
        # There does try     to be a dog      eating an apple.
        # THERE aux  control TO BE D emb_subj VP

        # emb_subj Zipf filter: closed-class curated words dominate; keep fallback for head/tail
        emb_subj_pool = filter_rows_for_active_zipf(self.safe_emb_subjs, "noun", fallback_on_empty=False)
        if len(emb_subj_pool) == 0:
            raise LexicalGapError("No xtail-compatible embedded subjects")

        for _ in range(self.max_sample_attempts):
            emb_subj = N_to_DP_mutate(choice(emb_subj_pool), determiner=False)
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
                # Curated control verbs — fallback_on_empty=True so xtail falls back to full curated list
                control_candidates = table_intersect1d(self.control_verbs, agree_verbs)
                control_row = choose_row_for_active_zipf(
                    control_candidates,
                    "verb",
                    fallback_on_empty=True,
                    error_message="No compatible control-subject verbs for agreement class",
                )
                aux = return_aux(control_row, emb_subj, allow_negated=allow_negated)
                control = control_row[0]
            else:
                control_row = choose_row_for_active_zipf(
                    self.control_pred_rows,
                    "adjective",
                    fallback_on_empty=True,
                    error_message="No compatible control-subject adjectives",
                ) if overlay_enabled() and len(self.control_pred_rows) > 0 else None
                control = control_row[0] if control_row is not None else choice(self.control_preds)
                aux = return_copula(emb_subj, allow_negated=allow_negated)

            if verbal_predicate:
                # Keep the good-side verbal predicate in the same inflectional shape as the sampled bad-side control verb.
                raising_candidates = rows_matching_inflection(self.raising_verbs, control_row)
                raising = choose_row_for_active_zipf(
                    raising_candidates,
                    "verb",
                    fallback_on_empty=True,
                    error_message="No compatible subject-raising verbs for auxiliary/agreement class",
                )[0]
            else:
                raising_row = choose_row_for_active_zipf(
                    self.raising_pred_rows,
                    "adjective",
                    fallback_on_empty=True,
                    error_message="No compatible subject-raising adjectives",
                ) if overlay_enabled() and len(self.raising_pred_rows) > 0 else None
                raising = raising_row[0] if raising_row is not None else choice(self.raising_preds)

            # Embedded -ing verb: Zipf-filter with fallback=False; retry on empty
            try:
                V = choose_row_for_active_zipf(
                    get_matched_by(emb_subj, "arg_1", all_ing_verbs),
                    "verb",
                    fallback_on_empty=False,
                    error_message="No compatible existential_there_subject_raising embedded verb",
                )
            except LexicalGapError:
                continue
            args = verb_args_from_verb(V, subj=emb_subj, allow_negated=allow_negated)
            VP = V_to_VP_mutate(V, args=args, aux=False)
            break
        else:
            raise LexicalGapError("No xtail-compatible existential_there_subject_raising sample after bounded retries")

        data = {
            "sentence_good": self._surface_sentence(aux, raising, D, emb_subj, VP),
            "sentence_bad": self._surface_sentence(aux, control, D, emb_subj, VP),
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
