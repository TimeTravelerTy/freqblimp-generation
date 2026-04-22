from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice, uniform_choice
from utils.vocab_sets import *

from generation_projects.blimp.overlay_guards import (
    filter_rows_for_active_zipf,
    finite_clause_embedding_verb_rows,
)

class BindingGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax_semantics",
                         linguistics="binding",
                         uid="principle_A_domain_3",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=True)
        self.all_gendered_singular_nouns = get_all("sg", "1", all_gendered_nouns)
        self.all_safe_gendered_nouns = np.setdiff1d(self.all_gendered_singular_nouns, all_relational_nouns)
        embedding_verbs = finite_clause_embedding_verb_rows()
        self.all_sing_embedding_verbs = np.union1d(
            get_all_conjunctive([("pres", "1"), ("3sg", "1")], embedding_verbs),
            get_all("bare", "1", embedding_verbs),
        )
        self.all_sing_refl_preds = np.union1d(get_all_conjunctive([("pres", "1"), ("3sg", "1")], all_refl_preds), get_all("bare", "1", all_refl_preds))
        self.max_sample_attempts = 64

    def sample(self):
        # John thinks Mary saw      herself.
        # N1   V1     N2   Vembed   refl_match
        # Mary thinks  John saw     herself.
        # N2   V1      N1   Vembed  refl_match

        for _ in range(self.max_sample_attempts):
            V1 = uniform_choice(self.all_sing_embedding_verbs)
            Vembed = uniform_choice(self.all_sing_refl_preds)
            n1_pool = filter_rows_for_active_zipf(get_matches_of(V1, "arg_1", self.all_safe_gendered_nouns), "noun")
            if len(n1_pool) == 0:
                continue
            N1 = N_to_DP_mutate(choice(n1_pool))
            refl_mismatch_pool = get_matched_by(N1, "arg_1", all_reflexives)
            if len(refl_mismatch_pool) == 0:
                continue
            refl_mismatch = choice(refl_mismatch_pool)
            n2_pool = filter_rows_for_active_zipf(get_matches_of(Vembed, "arg_1", self.all_safe_gendered_nouns), "noun")
            if len(n2_pool) == 0:
                continue
            N2 = choice(n2_pool)
            n_tries = 0
            while is_match_disj(N2, refl_mismatch["arg_1"]) and n_tries < 10:
                N2 = choice(n2_pool)
                n_tries += 1
            if n_tries == 10:
                continue
            refl_match_pool = get_matched_by(N2, "arg_1", all_reflexives)
            if len(refl_match_pool) == 0:
                continue
            refl_match = choice(refl_match_pool)
            N2 = N_to_DP_mutate(N2)
            V1 = conjugate(V1, N1)
            Vembed = conjugate(Vembed, N2)
            break
        else:
            raise LexicalGapError("No principle_A_domain_3 sample found after bounded retries")

        data = {
            "sentence_good": "%s %s %s %s %s." % (N1[0], V1[0], N2[0], Vembed[0], refl_match[0]),
            "sentence_bad": "%s %s %s %s %s." % (N2[0], V1[0], N1[0], Vembed[0], refl_match[0]),
            "two_prefix_prefix_good": "%s %s %s %s" % (N1[0], V1[0], N2[0], Vembed[0]),
            "two_prefix_prefix_bad": "%s %s %s %s" % (N2[0], V1[0], N1[0], Vembed[0]),
            "two_prefix_word": refl_match[0]
        }
        return data, data["sentence_good"]


def build_generator():
    return BindingGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
