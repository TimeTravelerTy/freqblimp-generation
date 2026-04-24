from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice, uniform_choice
from utils.exceptions import LexicalGapError

from generation_projects.blimp.overlay_guards import (
    clausal_it_adjective_rows,
    choose_matching_row,
    choose_row_for_active_zipf,
    control_object_verb_rows,
    filter_plural_looking_singular_nouns,
    object_raising_verb_rows,
    rows_matching_inflection,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax_semantics",
                         linguistics="control_raising",
                         uid="expletive_it_object_raising",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)
        self.clause_embedding_adjectives = clausal_it_adjective_rows()
        self.raising_verbs = object_raising_verb_rows()
        self.control_verbs = control_object_verb_rows()
        self.safe_nominals = filter_plural_looking_singular_nouns(all_nominals)
        self.safe_clause_verbs = table_union1d(
            get_all("category", "S\\NP", all_verbs),
            get_all("category", "(S\\NP)/NP", all_verbs),
        )
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
        subj_pool = filter_plural_looking_singular_nouns(
            get_matches_of(V_raise, "arg_1", get_matches_of(V_control, "arg_1"))
        )
        self._subject_pool_cache[key] = subj_pool
        return subj_pool

    def _make_safe_clause(self):
        for _ in range(128):
            try:
                V_emb = choose_row_for_active_zipf(
                    self.safe_clause_verbs,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No regime-compatible expletive-it embedded verb",
                )
                Subj = N_to_DP_mutate(choose_matching_row(
                    V_emb,
                    "arg_1",
                    self.safe_nominals,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible expletive-it embedded subject",
                ))
                Aux = return_aux(V_emb, Subj)
                pieces = [Subj[0], Aux[0], V_emb[0]]
                if V_emb["category"] == "(S\\NP)/NP":
                    Obj = N_to_DP_mutate(choose_matching_row(
                        V_emb,
                        "arg_2",
                        self.safe_nominals,
                        "noun",
                        fallback_on_empty=False,
                        error_message="No regime-compatible expletive-it embedded object",
                    ))
                    pieces.append(Obj[0])
                return remove_extra_whitespace(" ".join(pieces))
            except LexicalGapError:
                continue
        raise LexicalGapError("No regime-compatible expletive-it embedded clause found")

    def sample(self):
        # John   may        consider it to be unfortunate that Bill has left.
        # m_subj Aux_raise  V_raise  IT TO BE Adj         THAT sentence
        # John   may          persuade   it to be unfortunate that Bill has left.
        # m_subj Aux_control  V_control  IT TO BE Adj         THAT sentence

        if not self.compatible_pairs:
            raise LexicalGapError("No compatible raising/control expletive-it pairs found")

        subj_pool = None
        for _ in range(min(len(self.compatible_pairs), 128)):
            V_raise, V_control = uniform_choice(self.compatible_pairs)
            subj_pool = self._subject_pool_for_pair(V_raise, V_control)
            if len(subj_pool) > 0:
                break
        else:
            raise LexicalGapError("No compatible raising/control expletive-it subject pool found")

        m_subj = N_to_DP_mutate(choose_row_for_active_zipf(
            subj_pool,
            "noun",
            fallback_on_empty=False,
            error_message="No regime-compatible expletive-it matrix subject",
        ))
        Aux = return_aux(V_raise, m_subj)
        Adj = choose_row_for_active_zipf(
            self.clause_embedding_adjectives,
            "adjective",
            fallback_on_empty=True,
            minimum_candidates=10,
            error_message="No regime-compatible expletive-it adjective",
        )
        sentence = self._make_safe_clause()

        data = {
            "sentence_good": "%s %s %s it to be %s that %s." % (m_subj[0], Aux[0], V_raise[0], Adj[0], sentence),
            "sentence_bad": "%s %s %s it to be %s that %s." % (m_subj[0], Aux[0], V_control[0], Adj[0], sentence),
            "two_prefix_prefix_good": "%s %s %s it to be" % (m_subj[0], Aux[0], V_raise[0]),
            "two_prefix_prefix_bad": "%s %s %s it to be" % (m_subj[0], Aux[0], V_control[0]),
            "two_prefix_word": Adj[0]
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
