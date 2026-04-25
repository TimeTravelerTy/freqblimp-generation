from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from functools import reduce
from utils.vocab_sets import *

from generation_projects.blimp.overlay_guards import choose_row_for_active_zipf, filter_rows_for_active_zipf

REFLEXIVE_PREDICATE_EXPRESSIONS = (
    "abandon", "abandons", "abandoned", "abandoning",
    "admire", "admires", "admired", "admiring",
    "blame", "blames", "blamed", "blaming",
    "brand", "brands", "branded", "branding",
    "defend", "defends", "defended", "defending",
    "describe", "describes", "described", "describing",
    "help", "helps", "helped", "helping",
    "include", "includes", "included", "including",
    "introduce", "introduces", "introduced", "introducing",
    "measure", "measures", "measured", "measuring",
    "pay", "pays", "paid", "paying",
    "please", "pleases", "pleased", "pleasing",
    "post", "posts", "posted", "posting",
    "slice", "slices", "sliced", "slicing",
    "trust", "trusts", "trusted", "trusting",
    "watch", "watches", "watched", "watching",
)


class AnaphorGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(
            field="morphology",
            linguistics="anaphor_agreement",
            uid="anaphor_gender_agreement",
            simple_lm_method=True,
            one_prefix_method=True,
            two_prefix_method=False,
            lexically_identical=False
        )
        self.all_safe_nouns = np.setdiff1d(all_singular_nouns, all_singular_neuter_animate_nouns)
        self.all_singular_reflexives = reduce(np.union1d, (get_all("expression", "himself"),
                                                           get_all("expression", "herself"),
                                                           get_all("expression", "itself")))
        self.all_reflexive_matchable_safe_nouns = reduce(
            table_union1d,
            (get_matches_of(reflexive, "arg_1", self.all_safe_nouns) for reflexive in self.all_singular_reflexives),
        )
        self.noun_pool = filter_rows_for_active_zipf(
            self.all_reflexive_matchable_safe_nouns,
            "noun",
            minimum_candidates=1024,
        )
        predicate_rows = [
            rows for rows in (get_all("expression", expression, all_transitive_verbs)
                              for expression in REFLEXIVE_PREDICATE_EXPRESSIONS)
            if len(rows) > 0
        ]
        self.reflexive_predicates = reduce(table_union1d, predicate_rows)

    def sample(self):
        # John knows himself
        # N1   V1    refl_match
        # John knows itself
        # N1   V1    refl_mismatch

        for _ in range(128):
            V1 = choose_row_for_active_zipf(
                self.reflexive_predicates,
                "verb",
                fallback_on_empty=True,
                minimum_candidates=8,
            )
            noun_candidates = get_matches_of(V1, "arg_1", get_matches_of(V1, "arg_2", self.noun_pool))
            if len(noun_candidates) > 0:
                break
        N1 = N_to_DP_mutate(choice(noun_candidates))
        refl_match = choice(get_matched_by(N1, "arg_1", all_reflexives))
        refl_mismatch = choice(np.setdiff1d(self.all_singular_reflexives, [refl_match]))

        V1 = conjugate(V1, N1)

        data = {
            "sentence_good": "%s %s %s." % (N1[0], V1[0], refl_match[0]),
            "sentence_bad": "%s %s %s." % (N1[0], V1[0], refl_mismatch[0]),
            "one_prefix_prefix": "%s %s" % (N1[0], V1[0]),
            "one_prefix_word_good": refl_match[0],
            "one_prefix_word_bad": refl_mismatch[0]
        }
        return data, data["sentence_good"]



def build_generator():
    return AnaphorGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
