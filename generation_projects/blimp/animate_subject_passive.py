from utils import data_generator
from utils.conjugate import *
from utils.constituent_building import *
from utils.randomize import choice
from utils.string_utils import string_beautify
from functools import reduce
from utils.vocab_sets import *
from utils.exceptions import LexicalGapError

from generation_projects.blimp.overlay_guards import (
    choose_matching_row,
    choose_row_for_active_zipf,
    verbs_with_argument_slots,
)

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="animate_subject_passive",
                         simple_lm_method=True,
                         one_prefix_method=True,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.all_inanim_subj_allowing_verbs = verbs_with_argument_slots(all_inanimate_nouns, all_nouns, all_transitive_verbs)
        self.all_anim_subj_allowing_verbs = verbs_with_argument_slots(all_animate_nouns, all_nouns, all_transitive_verbs)
        self.all_anim_subj_verbs = table_setdiff1d(self.all_anim_subj_allowing_verbs, self.all_inanim_subj_allowing_verbs)
        self.dets = ['the', 'some']
        self.location_nouns = get_all("locale", "1")
        self.nonlocation_commonnouns = table_setdiff1d(all_common_nouns, self.location_nouns)
        self.max_sample_attempts = 512

    def sample(self):
        # The boy was talked to by the woman
        # N1      cop V1        by det N2_good
        # The boy was talked to by the car
        # N1      cop V1        by det N2_bad

        for _ in range(self.max_sample_attempts):
            try:
                V1 = choose_row_for_active_zipf(
                    get_all('en', '1', self.all_anim_subj_verbs),
                    "verb",
                    fallback_on_empty=False,
                    error_message="No xtail-compatible animate-subject passive verbs",
                )
                N1 = N_to_DP_mutate(choose_matching_row(
                    V1,
                    'arg_2',
                    all_nouns,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible passive subject",
                ))
                cop = return_copula(N1)
                det = choice(self.dets)
                N2_good = choose_row_for_active_zipf(
                    get_all("animate", "1", get_matched_by(V1, "arg_1", all_common_nouns)),
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible animate passive agent",
                )
                N2_bad = choose_row_for_active_zipf(
                    get_all("animate", "0", self.nonlocation_commonnouns),
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible inanimate passive agent",
                )
                break
            except LexicalGapError:
                continue
        else:
            raise LexicalGapError("No xtail-compatible animate_subject_passive pair found after bounded retries")

        data = {
            "sentence_good": "%s %s %s by %s %s." % (N1[0], cop[0], V1[0], det, N2_good[0]),
            "sentence_bad": "%s %s %s by %s %s." % (N1[0], cop[0], V1[0], det, N2_bad[0]),
            "one_prefix_prefix": "%s %s %s by %s" % (N1[0], cop[0], V1[0], det),
            "one_prefix_word_good": N2_good[0],
            "one_prefix_word_bad": N2_bad[0]
        }
        return data, data["sentence_good"]


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
