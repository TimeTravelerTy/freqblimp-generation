from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.vocab_sets import *
from utils.exceptions import LexicalGapError

from generation_projects.blimp.overlay_guards import (
    animate_subject_transitive_verb_rows,
    choose_matching_row,
    choose_row_for_active_zipf,
    choose_row_for_active_zipf_by_source_lemma,
    exclude_source_lemmas_present_in,
)

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="animate_subject_trans",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)
        bad_subject_pool = get_all("physical", "1", get_all("animate", "0", all_common_nouns))
        bad_subject_pool = table_setdiff1d(bad_subject_pool, get_all("animal", "1", bad_subject_pool))
        bad_subject_pool = table_setdiff1d(bad_subject_pool, get_all("person", "1", bad_subject_pool))
        self.bad_subject_pool = exclude_source_lemmas_present_in(bad_subject_pool, all_animate_nouns)
        self.all_anim_subj_verbs = animate_subject_transitive_verb_rows()
        self.dets = ['the', 'some']
        self.max_sample_attempts = 512

    def sample(self):
        # John      talked to the boy
        # N1_good   V1        N2
        # The table talked to the boy
        # N1_bad    V1        N2

        for _ in range(self.max_sample_attempts):
            try:
                V1 = choose_row_for_active_zipf_by_source_lemma(
                    self.all_anim_subj_verbs,
                    "verb",
                    fallback_on_empty=False,
                    error_message="No xtail-compatible animate-subject transitive verbs",
                )
                N1_good = N_to_DP_mutate(choose_matching_row(
                    V1,
                    "arg_1",
                    all_animate_nouns,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible animate subject",
                ))
                if N1_good['sg'] == '1':
                    N1_bad = N_to_DP_mutate(choose_row_for_active_zipf(
                        get_all('sg', '1', self.bad_subject_pool),
                        "noun",
                        fallback_on_empty=False,
                        error_message="No regime-compatible inanimate singular subject",
                    ))
                elif N1_good['pl'] == '1':
                    N1_bad = N_to_DP_mutate(choose_row_for_active_zipf(
                        get_all('pl', '1', self.bad_subject_pool),
                        "noun",
                        fallback_on_empty=False,
                        error_message="No regime-compatible inanimate plural subject",
                    ))
                else:
                    raise ValueError("Subject must be singular or plural.")
                N2 = N_to_DP_mutate(choose_matching_row(
                    V1,
                    "arg_2",
                    all_nouns,
                    "noun",
                    fallback_on_empty=False,
                    error_message="No regime-compatible object noun",
                ))
                V1_conj = conjugate(V1, N1_good)
                break
            except LexicalGapError:
                continue
        else:
            raise LexicalGapError("No xtail-compatible animate_subject_trans pair found after bounded retries")

        data = {
            "sentence_good": "%s %s %s." % (N1_good[0], V1_conj[0], N2[0]),
            "sentence_bad": "%s %s %s." % (N1_bad[0], V1_conj[0], N2[0]),
            "two_prefix_prefix_good": "%s" % (N1_good[0]),
            "two_prefix_prefix_bad": "%s" % (N1_bad[0]),
            "two_prefix_word": V1_conj[0]
        }
        return data, data["sentence_good"]


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
