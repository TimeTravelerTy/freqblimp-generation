from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import dp_buildable_nominal_rows, filter_rows_for_active_zipf

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="intransitive",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)

        self.intransitive_verbs = get_all_conjunctive([("category", "S\\NP"), ("strict_intrans", "1")], all_verbs)
        self.strict_transitive = get_all("strict_trans", "1", all_transitive_verbs)
        intransitive_exprs = set(map(str, self.intransitive_verbs["expression"]))
        keep_mask = ~np.isin(np.asarray(self.strict_transitive["expression"], dtype=str), list(intransitive_exprs))
        self.strict_transitive = self.strict_transitive[keep_mask]
        self.safe_subjects = dp_buildable_nominal_rows()
        self.max_sample_attempts = 512

    def _surface_sentence(self, subj, aux, verb):
        text = remove_extra_whitespace(f"{subj[0]} {aux[0]} {verb[0]}")
        return text[0].upper() + text[1:] + "."

    def sample(self):
        # The bear has slept.
        # Subj     Aux V_intrans
        # The bear has injured.
        # Subj     Aux V_trans

        verb_pool = filter_rows_for_active_zipf(self.intransitive_verbs, "verb", fallback_on_empty=False)
        if len(verb_pool) == 0:
            raise LexicalGapError("No regime-compatible intransitive verbs")

        for _ in range(self.max_sample_attempts):
            V_intrans = choice(verb_pool)
            subj_pool = filter_rows_for_active_zipf(
                get_matches_of(V_intrans, "arg_1", self.safe_subjects), "noun", fallback_on_empty=False
            )
            if len(subj_pool) == 0:
                continue
            Subj = N_to_DP_mutate(choice(subj_pool))
            Aux = return_aux(V_intrans, Subj)
            bad_pool = filter_rows_for_active_zipf(
                get_matched_by(Subj, "arg_1", get_matches_of(Aux, "arg_2", self.strict_transitive)),
                "verb",
                fallback_on_empty=False,
            )
            if len(bad_pool) == 0:
                continue
            V_trans = choice(bad_pool)
            break
        else:
            raise LexicalGapError("No regime-compatible intransitive pair found after bounded retries")

        data = {
            "sentence_good": self._surface_sentence(Subj, Aux, V_intrans),
            "sentence_bad": self._surface_sentence(Subj, Aux, V_trans),
        }
        return data, data["sentence_good"]


def build_generator():
    return Generator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
