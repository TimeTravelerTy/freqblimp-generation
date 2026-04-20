from utils import data_generator
from utils.constituent_building import *
from utils.conjugate import *
from utils.randomize import choice
from utils.vocab_sets import *

from generation_projects.blimp.overlay_guards import filter_rows_for_active_zipf

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="syntax",
                         linguistics="argument_structure",
                         uid="animate_subject_trans",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)
        self.all_inanim_subj_allowing_verbs = get_matched_by(choice(all_inanimate_nouns), "arg_1", all_transitive_verbs)
        self.all_anim_subj_allowing_verbs = get_matched_by(choice(all_animate_nouns), "arg_1", all_transitive_verbs)
        self.all_anim_subj_verbs = np.setdiff1d(self.all_anim_subj_allowing_verbs, self.all_inanim_subj_allowing_verbs)
        self.dets = ['the', 'some']
        self.max_sample_attempts = 512

    def sample(self):
        # John      talked to the boy
        # N1_good   V1        N2
        # The table talked to the boy
        # N1_bad    V1        N2

        verb_pool = filter_rows_for_active_zipf(self.all_anim_subj_verbs, "verb", fallback_on_empty=False)
        if len(verb_pool) == 0:
            raise LexicalGapError("No xtail-compatible animate-subject transitive verbs")

        for _ in range(self.max_sample_attempts):
            V1 = choice(verb_pool)
            N1_good_candidates = filter_rows_for_active_zipf(
                get_matches_of(V1, "arg_1", all_nouns), "noun", fallback_on_empty=False
            )
            if len(N1_good_candidates) == 0:
                continue
            N1_good = N_to_DP_mutate(choice(N1_good_candidates))
            if N1_good['sg'] == '1':
                N1_bad_candidates = filter_rows_for_active_zipf(
                    get_all('sg', '1', all_inanimate_nouns), "noun", fallback_on_empty=False
                )
            elif N1_good['pl'] == '1':
                N1_bad_candidates = filter_rows_for_active_zipf(
                    get_all('pl', '1', all_inanimate_nouns), "noun", fallback_on_empty=False
                )
            else:
                raise ValueError("Subject must be singular or plural.")
            if len(N1_bad_candidates) == 0:
                continue
            N1_bad = N_to_DP_mutate(choice(N1_bad_candidates))
            N2_candidates = filter_rows_for_active_zipf(
                get_matches_of(V1, "arg_2", all_nouns), "noun", fallback_on_empty=False
            )
            if len(N2_candidates) == 0:
                continue
            N2 = N_to_DP_mutate(choice(N2_candidates))
            V1_conj = conjugate(V1, N1_good)
            break
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
