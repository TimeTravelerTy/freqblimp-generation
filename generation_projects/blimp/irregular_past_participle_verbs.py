from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice, uniform_choice
from utils.vocab_sets import *

from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="morphology",
                         linguistics="irregular_forms",
                         uid="irregular_past_participle_verbs",
                         simple_lm_method=True,
                         one_prefix_method=True,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.all_trans_en_verbs = get_all("special_en_form", "1", all_transitive_verbs)
        self.all_intrans_en_verbs = get_all("special_en_form", "1", all_intransitive_verbs)

    def _has_distinct_past_participle(self, verb):
        family = get_all("root", verb["root"])
        past = get_all("past", "1", family)
        participle = get_all("en", "1", family)
        if len(past) == 0 or len(participle) == 0:
            return False
        return str(past[0][0]).strip().lower() != str(participle[0][0]).strip().lower()

    def sample(self):
        # John ate    the pie
        # N1   V_past     N2
        # John eaten the pie
        # N1   V_en      N2

        x = random.random()
        if x < 1 / 2:
            V_base = uniform_choice(self.all_trans_en_verbs)
            while not self._has_distinct_past_participle(V_base):
                V_base = uniform_choice(self.all_trans_en_verbs)
            N2 = N_to_DP_mutate(uniform_choice(filter_rows_for_active_zipf(get_matches_of(V_base, "arg_2", all_nouns), "noun")))
        else:
            V_base = uniform_choice(self.all_intrans_en_verbs)
            while not self._has_distinct_past_participle(V_base):
                V_base = uniform_choice(self.all_intrans_en_verbs)
            N2 = " "

        Verbs = get_all("root", V_base["root"])
        V_past = get_all("past", "1", Verbs)
        V_en = get_all("en", "1", Verbs)
        N1 = N_to_DP_mutate(uniform_choice(filter_rows_for_active_zipf(get_matches_of(V_base, "arg_1", all_nouns), "noun")))

        data = {
            "sentence_good": "%s %s %s." % (N1[0], V_past[0][0], N2[0]),
            "sentence_bad": "%s %s %s." % (N1[0], V_en[0][0], N2[0]),
            "one_prefix_prefix": "%s" % (N1[0]),
            "one_prefix_word_good": V_past[0][0],
            "one_prefix_word_bad": V_en[0][0]
        }
        return data, data["sentence_good"]


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
