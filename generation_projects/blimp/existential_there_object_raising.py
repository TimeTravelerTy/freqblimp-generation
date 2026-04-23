from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice, uniform_choice
from utils.exceptions import LexicalGapError
from functools import reduce

from generation_projects.blimp.overlay_guards import (
    choose_row_for_active_zipf,
    control_object_verb_rows,
    object_raising_verb_rows,
    rows_matching_inflection,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax_semantics",
                         linguistics="control_raising",
                         uid="existential_there_object_raising",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)
        good_quantifiers_sg_str = ["a", "an", ""]
        good_quantifiers_pl_str = ["no", "some", "few", "fewer than three", "more than three", "many", "a lot of", ""]
        self.good_quantifiers_sg = reduce(np.union1d, [get_all("expression", s, all_determiners) for s in good_quantifiers_sg_str])
        self.good_quantifiers_pl = reduce(np.union1d, [get_all("expression", s, all_determiners) for s in good_quantifiers_pl_str])
        bad_emb_subjs = reduce(np.union1d, (all_relational_poss_nouns, all_proper_names, get_all("category", "NP")))
        self.safe_emb_subjs = np.setdiff1d(all_nominals, bad_emb_subjs)
        self.raising_verbs = object_raising_verb_rows()
        self.control_verbs = control_object_verb_rows()
        self.compatible_pairs = []
        self._subject_pool_cache = {}
        for V_raise in self.raising_verbs:
            compatible_controls = rows_matching_inflection(self.control_verbs, V_raise)
            for V_control in compatible_controls:
                self.compatible_pairs.append((V_raise, V_control))
        self.compatible_pairs = tuple(self.compatible_pairs)

    def _subject_pool_for_pair(self, V_raise, V_control):
        key = (str(V_raise["arg_1"]), str(V_control["arg_1"]))
        cached = self._subject_pool_cache.get(key)
        if cached is not None:
            return cached
        subj_pool = get_matches_of(V_raise, "arg_1", get_matches_of(V_control, "arg_1"))
        self._subject_pool_cache[key] = subj_pool
        return subj_pool

    def sample(self):
        # John   believed there to be a party    happening
        # m_subj V_raise  THERE TO BE D emb_subj VP
        # John   persuaded there to be a party    happening
        # m_subj V_control THERE TO BE D emb_subj VP

        if not self.compatible_pairs:
            raise LexicalGapError("No compatible raising/control object-raising pairs found")

        subj_pool = None
        for _ in range(min(len(self.compatible_pairs), 128)):
            V_raise, V_control = uniform_choice(self.compatible_pairs)
            subj_pool = self._subject_pool_for_pair(V_raise, V_control)
            if len(subj_pool) > 0:
                break
        else:
            raise LexicalGapError("No compatible raising/control object-raising subject pool found")

        m_subj = N_to_DP_mutate(choose_row_for_active_zipf(
            subj_pool,
            "noun",
            fallback_on_empty=False,
            error_message="No regime-compatible existential_there_object_raising matrix subject",
        ))

        Aux = return_aux(V_raise, m_subj)

        emb_subj = N_to_DP_mutate(choose_row_for_active_zipf(
            self.safe_emb_subjs,
            "noun",
            fallback_on_empty=False,
            error_message="No regime-compatible existential_there_object_raising embedded subject",
        ), determiner=False)
        D = choice(get_matched_by(emb_subj, "arg_1", self.good_quantifiers_sg)) \
            if emb_subj["sg"] == "1" \
            else choice(get_matched_by(emb_subj, "arg_1", self.good_quantifiers_pl))
        V = choose_row_for_active_zipf(
            get_matched_by(emb_subj, "arg_1", all_ing_verbs),
            "verb",
            fallback_on_empty=False,
            error_message="No regime-compatible existential_there_object_raising embedded verb",
        )
        allow_negated = D[0] != "no" and D[0] != "some"
        args = verb_args_from_verb(V, subj=emb_subj, allow_negated=allow_negated)
        VP = V_to_VP_mutate(V, args=args, aux=False)

        data = {
            "sentence_good": "%s %s %s there to be %s %s %s." % (m_subj[0], Aux[0], V_raise[0], D[0], emb_subj[0], VP[0]),
            "sentence_bad": "%s %s %s there to be %s %s %s." % (m_subj[0], Aux[0], V_control[0], D[0], emb_subj[0], VP[0]),
            "two_prefix_prefix_good": "%s %s %s" % (m_subj[0], Aux[0], V_raise[0]),
            "two_prefix_prefix_bad": "%s %s %s" % (m_subj[0], Aux[0], V_control[0]),
            "two_prefix_word": "there"
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
