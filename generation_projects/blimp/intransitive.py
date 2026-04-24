from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice

from generation_projects.blimp.overlay_guards import (
    choose_matching_row,
    choose_matched_by_row,
    choose_row_for_active_zipf,
    dp_buildable_nominal_rows,
    filter_plural_looking_singular_nouns,
    pure_strict_intransitive_rows,
    pure_strict_transitive_rows,
)

class Generator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="intransitive",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=False,
                         lexically_identical=False)

        self.intransitive_verbs = pure_strict_intransitive_rows()
        self.strict_transitive = pure_strict_transitive_rows()
        intransitive_exprs = set(map(str, self.intransitive_verbs["expression"]))
        keep_mask = ~np.isin(np.asarray(self.strict_transitive["expression"], dtype=str), list(intransitive_exprs))
        self.strict_transitive = self.strict_transitive[keep_mask]
        self.safe_subjects = filter_plural_looking_singular_nouns(dp_buildable_nominal_rows())
        self.max_sample_attempts = 512

    def _surface_sentence(self, subj, aux, verb):
        text = remove_extra_whitespace(f"{subj[0]} {aux[0]} {verb[0]}")
        return text[0].upper() + text[1:] + "."

    def sample(self):
        # The bear has slept.
        # Subj     Aux V_intrans
        # The bear has injured.
        # Subj     Aux V_trans

        for _ in range(self.max_sample_attempts):
            try:
                V_intrans = choose_row_for_active_zipf(
                    self.intransitive_verbs,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No regime-compatible intransitive verbs",
                )
                Subj = N_to_DP_mutate(choose_matching_row(
                    V_intrans,
                    "arg_1",
                    self.safe_subjects,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible intransitive subject",
                ))
                Aux = return_aux(V_intrans, Subj)
                V_trans = choose_matched_by_row(
                    Subj,
                    "arg_1",
                    get_matches_of(Aux, "arg_2", self.strict_transitive),
                    "verb",
                    fallback_on_empty=False,
                    error_message="No regime-compatible transitive contrast verb",
                )
                break
            except LexicalGapError:
                continue
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
