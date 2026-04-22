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
                         uid="principle_A_domain_2",
                         simple_lm_method=True,
                         one_prefix_method=True,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.all_safe_nouns = np.setdiff1d(all_nouns, all_singular_neuter_animate_nouns)
        self.embedding_verbs = finite_clause_embedding_verb_rows()
        self.max_sample_attempts = 64

    def sample(self):
        # John thinks Mary saw      herself.
        # N1   V1     N2   Vembed   refl_match
        # John thinks  Mary saw     himself.
        # N1   V1      N2   Vembed  refl_mismatch

        for _ in range(self.max_sample_attempts):
            V1 = uniform_choice(self.embedding_verbs)
            Vembed = uniform_choice(all_refl_preds)
            n1_pool = filter_rows_for_active_zipf(get_matches_of(V1, "arg_1", self.all_safe_nouns), "noun")
            if len(n1_pool) == 0:
                continue
            N1 = N_to_DP_mutate(choice(n1_pool))
            refl_mismatch_pool = get_matched_by(N1, "arg_1", all_reflexives)
            if len(refl_mismatch_pool) == 0:
                continue
            refl_mismatch = choice(refl_mismatch_pool)
            n2_pool = filter_rows_for_active_zipf(get_matches_of(Vembed, "arg_1", self.all_safe_nouns), "noun")
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
            raise LexicalGapError("No principle_A_domain_2 sample found after bounded retries")

        data = {
            "sentence_good": "%s %s %s %s %s." % (N1[0], V1[0], N2[0], Vembed[0], refl_match[0]),
            "sentence_bad": "%s %s %s %s %s." % (N1[0], V1[0], N2[0], Vembed[0], refl_mismatch[0]),
            "one_prefix_prefix": "%s %s %s %s" % (N1[0], V1[0], N2[0], Vembed[0]),
            "one_prefix_word_good": refl_match[0],
            "one_prefix_word_bad": refl_mismatch[0]
        }
        return data, data["sentence_good"]


def build_generator():
    return BindingGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
