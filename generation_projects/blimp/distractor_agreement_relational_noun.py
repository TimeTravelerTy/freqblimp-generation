import random

from utils import data_generator
from utils.constituent_building import *
from utils.exceptions import LexicalGapError
from utils.randomize import choice, uniform_choice
from utils.vocab_sets import *

from generation_projects.blimp.overlay_guards import (
    filter_rows_for_active_zipf,
    mismatching_nonpast_agreement_form,
    non_past_verb_rows,
)

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="morphology",
                         linguistics="subject_verb_agreement",
                         uid="distractor_agreement_relational_noun",
                         simple_lm_method=True,
                         one_prefix_method=True,
                         two_prefix_method=False,
                         lexically_identical=False)
        self.all_reg_nouns = get_all_conjunctive([("noun", "1"), ("irrpl", "")])
        self.safe_subjs = get_all("category", "N/NP", self.all_reg_nouns)
        self.safe_verbs = np.intersect1d(non_past_verb_rows(), all_non_recursive_verbs)
        self.present_verbs = get_all("pres", "1", self.safe_verbs)
        self.ing_verbs = get_all("ing", "1", self.safe_verbs)
        self.en_verbs = get_all("en", "1", self.safe_verbs)
        self.progressive_aux_pairs = [
            ("is", "are"),
            ("isn't", "aren't"),
            ("was", "were"),
            ("wasn't", "weren't"),
        ]
        self.perfect_aux_pairs = [
            ("has", "have"),
            ("hasn't", "haven't"),
        ]

    def _opposite_number_arg(self, subj):
        noun_space = all_plural_nouns if subj["sg"] == "1" else all_singular_nouns
        return N_to_DP_mutate(
            choice(
                filter_rows_for_active_zipf(
                    get_matches_of(subj, "arg_1", noun_space),
                    "noun",
                    fallback_on_empty=False,
                )
            )
        )

    def _args_for(self, verb, subj):
        return join_args(
            verb_args_from_verb(
                verb,
                subj=subj,
                aux=False,
                allow_modal=False,
                allow_negated=False,
            )["args"]
        )

    def _format_sentence(self, prefix, first_word, verb, args):
        pieces = [prefix, first_word]
        if verb:
            pieces.append(verb[0])
        if args:
            pieces.append(args)
        text = remove_extra_whitespace(" ".join(pieces))
        return text[0].upper() + text[1:] + "."

    def sample(self):
        for _ in range(256):
            subj = uniform_choice(self.safe_subjs)
            D = choice(get_matched_by(subj, "arg_1", all_very_common_dets))
            try:
                S_arg = self._opposite_number_arg(subj)
            except LexicalGapError:
                continue
            prefix = f"{D[0]} {subj[0]} {S_arg[0]}"
            subject_number = "sg" if subj["sg"] == "1" else "pl"
            family = random.choice(("lexical", "progressive", "perfect"))

            try:
                if family == "lexical":
                    verb_seed = choice(
                        filter_rows_for_active_zipf(
                            get_matched_by(subj, "arg_1", self.present_verbs),
                            "verb",
                            fallback_on_empty=False,
                        )
                    )
                    other_form = mismatching_nonpast_agreement_form(verb_seed)
                    if verb_seed["3sg"] == "1":
                        singular_verb, plural_verb = verb_seed, other_form
                    else:
                        singular_verb, plural_verb = other_form, verb_seed
                    good_verb = singular_verb if subject_number == "sg" else plural_verb
                    bad_verb = plural_verb if subject_number == "sg" else singular_verb
                    args = self._args_for(good_verb, subj)
                    word_good = good_verb[0].split(" ")[0]
                    word_bad = bad_verb[0].split(" ")[0]
                    sentence_good = self._format_sentence(prefix, good_verb[0], None, args)
                    sentence_bad = self._format_sentence(prefix, bad_verb[0], None, args)
                elif family == "progressive":
                    verb = choice(
                        filter_rows_for_active_zipf(
                            get_matched_by(subj, "arg_1", self.ing_verbs),
                            "verb",
                            fallback_on_empty=False,
                        )
                    )
                    args = self._args_for(verb, subj)
                    singular_aux, plural_aux = random.choice(self.progressive_aux_pairs)
                    good_aux = singular_aux if subject_number == "sg" else plural_aux
                    bad_aux = plural_aux if subject_number == "sg" else singular_aux
                    word_good = good_aux
                    word_bad = bad_aux
                    sentence_good = self._format_sentence(prefix, good_aux, verb, args)
                    sentence_bad = self._format_sentence(prefix, bad_aux, verb, args)
                else:
                    verb = choice(
                        filter_rows_for_active_zipf(
                            get_matched_by(subj, "arg_1", self.en_verbs),
                            "verb",
                            fallback_on_empty=False,
                        )
                    )
                    args = self._args_for(verb, subj)
                    singular_aux, plural_aux = random.choice(self.perfect_aux_pairs)
                    good_aux = singular_aux if subject_number == "sg" else plural_aux
                    bad_aux = plural_aux if subject_number == "sg" else singular_aux
                    word_good = good_aux
                    word_bad = bad_aux
                    sentence_good = self._format_sentence(prefix, good_aux, verb, args)
                    sentence_bad = self._format_sentence(prefix, bad_aux, verb, args)
            except LexicalGapError:
                continue

            data = {
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "one_prefix_prefix": prefix,
                "one_prefix_word_good": word_good,
                "one_prefix_word_bad": word_bad,
            }
            return data, data["sentence_good"]

        raise LexicalGapError("No distractor_agreement_relational_noun sample found after bounded retries")


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
