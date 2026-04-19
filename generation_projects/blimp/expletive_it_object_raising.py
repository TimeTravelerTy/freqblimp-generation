from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice, uniform_choice
from utils.exceptions import LexicalGapError

from generation_projects.blimp.overlay_guards import (
    clausal_it_adjective_rows,
    control_object_verb_rows,
    object_raising_verb_rows,
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
        self.compatible_pairs = []
        for V_raise in self.raising_verbs:
            for V_control in self.control_verbs:
                subj_pool = get_matches_of(V_raise, "arg_1", get_matches_of(V_control, "arg_1"))
                if len(subj_pool) > 0:
                    self.compatible_pairs.append((V_raise, V_control, subj_pool))
        self.compatible_pairs = tuple(self.compatible_pairs)

    def sample(self):
        # John   may        consider it to be unfortunate that Bill has left.
        # m_subj Aux_raise  V_raise  IT TO BE Adj         THAT sentence
        # John   may          persuade   it to be unfortunate that Bill has left.
        # m_subj Aux_control  V_control  IT TO BE Adj         THAT sentence

        if not self.compatible_pairs:
            raise LexicalGapError("No compatible raising/control expletive-it pairs found")

        V_raise, V_control, subj_pool = uniform_choice(self.compatible_pairs)
        m_subj = N_to_DP_mutate(choice(subj_pool))
        Aux_raise = return_aux(V_raise, m_subj)
        Aux_control = return_aux(V_control, m_subj)
        Adj = choice(self.clause_embedding_adjectives)
        V_emb = choice(all_verbs)
        sentence = make_sentence_from_verb(V_emb)

        data = {
            "sentence_good": "%s %s %s it to be %s that %s." % (m_subj[0], Aux_raise[0], V_raise[0], Adj[0], sentence),
            "sentence_bad": "%s %s %s it to be %s that %s." % (m_subj[0], Aux_control[0], V_control[0], Adj[0], sentence),
            "two_prefix_prefix_good": "%s %s %s it to be" % (m_subj[0], Aux_raise[0], V_raise[0]),
            "two_prefix_prefix_bad": "%s %s %s it to be" % (m_subj[0], Aux_control[0], V_control[0]),
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
