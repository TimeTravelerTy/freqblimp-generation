import random

from utils import data_generator
from utils.constituent_building import *
from utils.exceptions import LexicalGapError
from utils.randomize import choice
from utils.vocab_sets import *

from generation_projects.blimp.overlay_guards import (
    build_agreement_safe_verbs,
    filter_rows_for_active_zipf,
    mismatching_nonpast_agreement_form,
)

class AgreementGenerator(data_generator.BenchmarkGenerator):
    def __init__(self):
        super().__init__(field="morphology",
                         linguistics="subject_verb_agreement",
                         uid="regular_plural_subject_verb_agreement_1",
                         simple_lm_method=True,
                         one_prefix_method=True,
                         two_prefix_method=False,
                         lexically_identical=False)
        safe_verbs = np.intersect1d(build_agreement_safe_verbs(), all_non_recursive_verbs)
        self.present_verbs = get_all("pres", "1", safe_verbs)
        self.ing_verbs = get_all("ing", "1", safe_verbs)
        self.en_verbs = get_all("en", "1", safe_verbs)
        singular_common = get_all_conjunctive([("category", "N"), ("sg", "1"), ("mass", "0")], all_common_nouns)
        plural_common = get_all_conjunctive([("category", "N"), ("pl", "1"), ("mass", "0")], all_common_nouns)
        self.singular_subjects = np.union1d(all_proper_names, singular_common)
        self.plural_subjects = plural_common
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

    def _subject_pool(self, verb, number):
        subject_space = self.singular_subjects if number == "sg" else self.plural_subjects
        return filter_rows_for_active_zipf(
            get_matches_of(verb, "arg_1", subject_space),
            "noun",
            fallback_on_empty=False,
        )

    def _pick_subject(self, verb, number):
        subj = widen_expression_field(choice(self._subject_pool(verb, number)))
        if subj["properNoun"] == "1" or subj["category"] == "NP":
            realized = subj
        else:
            realized = N_to_DP_mutate(subj)
        surface = remove_extra_whitespace(realized[0])
        realized[0] = surface[0].upper() + surface[1:]
        return realized

    def _args_for(self, verb, subj):
        args = verb_args_from_verb(
            verb,
            subj=subj,
            aux=False,
            allow_modal=False,
            allow_negated=False,
        )["args"]
        return join_args(args)

    def _format_sentence(self, subj, first_word, verb, args):
        pieces = [subj[0], first_word]
        if verb:
            pieces.append(verb[0])
        if args:
            pieces.append(args)
        return remove_extra_whitespace(" ".join(pieces)) + "."

    def sample(self):
        for _ in range(256):
            family = random.choice(("lexical", "progressive", "perfect"))
            number = random.choice(("sg", "pl"))

            try:
                if family == "lexical":
                    verb_seed = choice(filter_rows_for_active_zipf(self.present_verbs, "verb", fallback_on_empty=False))
                    other_form = mismatching_nonpast_agreement_form(verb_seed)
                    if verb_seed["3sg"] == "1":
                        singular_verb, plural_verb = verb_seed, other_form
                    else:
                        singular_verb, plural_verb = other_form, verb_seed
                    subj = self._pick_subject(singular_verb if number == "sg" else plural_verb, number)
                    args = self._args_for(singular_verb if number == "sg" else plural_verb, subj)
                    good_verb = singular_verb if number == "sg" else plural_verb
                    bad_verb = plural_verb if number == "sg" else singular_verb
                    word_good = good_verb[0].split(" ")[0]
                    word_bad = bad_verb[0].split(" ")[0]
                    sentence_good = self._format_sentence(subj, good_verb[0], None, args)
                    sentence_bad = self._format_sentence(subj, bad_verb[0], None, args)
                elif family == "progressive":
                    verb = choice(filter_rows_for_active_zipf(self.ing_verbs, "verb", fallback_on_empty=False))
                    subj = self._pick_subject(verb, number)
                    args = self._args_for(verb, subj)
                    singular_aux, plural_aux = random.choice(self.progressive_aux_pairs)
                    good_aux = singular_aux if number == "sg" else plural_aux
                    bad_aux = plural_aux if number == "sg" else singular_aux
                    word_good = good_aux
                    word_bad = bad_aux
                    sentence_good = self._format_sentence(subj, good_aux, verb, args)
                    sentence_bad = self._format_sentence(subj, bad_aux, verb, args)
                else:
                    verb = choice(filter_rows_for_active_zipf(self.en_verbs, "verb", fallback_on_empty=False))
                    subj = self._pick_subject(verb, number)
                    args = self._args_for(verb, subj)
                    singular_aux, plural_aux = random.choice(self.perfect_aux_pairs)
                    good_aux = singular_aux if number == "sg" else plural_aux
                    bad_aux = plural_aux if number == "sg" else singular_aux
                    word_good = good_aux
                    word_bad = bad_aux
                    sentence_good = self._format_sentence(subj, good_aux, verb, args)
                    sentence_bad = self._format_sentence(subj, bad_aux, verb, args)
            except LexicalGapError:
                continue

            data = {
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "one_prefix_prefix": subj[0],
                "one_prefix_word_good": word_good,
                "one_prefix_word_bad": word_bad,
            }
            return data, data["sentence_good"]

        raise LexicalGapError("No regular_plural_subject_verb_agreement_1 sample found after bounded retries")


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
