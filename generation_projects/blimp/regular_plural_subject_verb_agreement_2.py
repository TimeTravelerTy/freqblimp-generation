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
                         uid="regular_plural_subject_verb_agreement_2",
                         simple_lm_method=True,
                         one_prefix_method=False,
                         two_prefix_method=True,
                         lexically_identical=False)
        safe_verbs = np.intersect1d(build_agreement_safe_verbs(), all_non_recursive_verbs)
        self.present_verbs = get_all("pres", "1", safe_verbs)
        self.ing_verbs = get_all("ing", "1", safe_verbs)
        self.en_verbs = get_all("en", "1", safe_verbs)
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
        singular_pool = get_all_conjunctive(
            [("category", "N"), ("sg", "1"), ("mass", "0"), ("irrpl", "")],
            all_common_nouns,
        )
        self.singular_rows = []
        self.plural_map = {}
        for noun in singular_pool:
            plural_expression = self._expected_regular_plural(noun["expression"])
            if plural_expression is None or noun["pluralform"] != plural_expression:
                continue
            plural_rows = get_all_conjunctive(
                [("expression", plural_expression), ("pl", "1"), ("mass", "0")],
                all_common_nouns,
            )
            if len(plural_rows) == 0:
                continue
            self.singular_rows.append(noun.copy())
            self.plural_map[str(noun["expression"])] = plural_expression
        self.singular_rows = np.asarray(self.singular_rows, dtype=all_common_nouns.dtype)

    def _expected_regular_plural(self, expression):
        if " " in expression or "-" in expression or "'" in expression:
            return None
        if not expression.isalpha():
            return None
        if expression.endswith("y") and len(expression) > 1 and expression[-2] not in "aeiou":
            return expression[:-1] + "ies"
        if expression.endswith(("s", "x", "z", "ch", "sh")):
            return expression + "es"
        return expression + "s"

    def _matched_singular_nouns(self, verb):
        return filter_rows_for_active_zipf(
            get_matches_of(verb, "arg_1", self.singular_rows),
            "noun",
            fallback_on_empty=False,
        )

    def _args_for(self, verb):
        return join_args(
            verb_args_from_verb(
                verb,
                aux=False,
                allow_modal=False,
                allow_negated=False,
            )["args"]
        )

    def _format_sentence(self, subject, first_word, verb, args):
        pieces = [subject, first_word]
        if verb:
            pieces.append(verb[0])
        if args:
            pieces.append(args)
        return remove_extra_whitespace(" ".join(pieces)) + "."

    def sample(self):
        for _ in range(256):
            family = random.choice(("lexical", "progressive", "perfect"))
            good_number = random.choice(("sg", "pl"))

            try:
                if family == "lexical":
                    verb_seed = choice(filter_rows_for_active_zipf(self.present_verbs, "verb", fallback_on_empty=False))
                    other_form = mismatching_nonpast_agreement_form(verb_seed)
                    if verb_seed["3sg"] == "1":
                        singular_verb, plural_verb = verb_seed, other_form
                    else:
                        singular_verb, plural_verb = other_form, verb_seed
                    match_verb = singular_verb if good_number == "sg" else plural_verb
                    noun = choice(self._matched_singular_nouns(match_verb))
                    plural_expression = self.plural_map[str(noun["expression"])]
                    args = self._args_for(match_verb)
                    fixed_word = singular_verb[0] if good_number == "sg" else plural_verb[0]
                    sentence_good = self._format_sentence(
                        f"The {noun[0]}" if good_number == "sg" else f"The {plural_expression}",
                        fixed_word,
                        None,
                        args,
                    )
                    sentence_bad = self._format_sentence(
                        f"The {plural_expression}" if good_number == "sg" else f"The {noun[0]}",
                        fixed_word,
                        None,
                        args,
                    )
                elif family == "progressive":
                    verb = choice(filter_rows_for_active_zipf(self.ing_verbs, "verb", fallback_on_empty=False))
                    noun = choice(self._matched_singular_nouns(verb))
                    plural_expression = self.plural_map[str(noun["expression"])]
                    args = self._args_for(verb)
                    singular_aux, plural_aux = random.choice(self.progressive_aux_pairs)
                    fixed_word = singular_aux if good_number == "sg" else plural_aux
                    sentence_good = self._format_sentence(
                        f"The {noun[0]}" if good_number == "sg" else f"The {plural_expression}",
                        fixed_word,
                        verb,
                        args,
                    )
                    sentence_bad = self._format_sentence(
                        f"The {plural_expression}" if good_number == "sg" else f"The {noun[0]}",
                        fixed_word,
                        verb,
                        args,
                    )
                else:
                    verb = choice(filter_rows_for_active_zipf(self.en_verbs, "verb", fallback_on_empty=False))
                    noun = choice(self._matched_singular_nouns(verb))
                    plural_expression = self.plural_map[str(noun["expression"])]
                    args = self._args_for(verb)
                    singular_aux, plural_aux = random.choice(self.perfect_aux_pairs)
                    fixed_word = singular_aux if good_number == "sg" else plural_aux
                    sentence_good = self._format_sentence(
                        f"The {noun[0]}" if good_number == "sg" else f"The {plural_expression}",
                        fixed_word,
                        verb,
                        args,
                    )
                    sentence_bad = self._format_sentence(
                        f"The {plural_expression}" if good_number == "sg" else f"The {noun[0]}",
                        fixed_word,
                        verb,
                        args,
                    )
            except LexicalGapError:
                continue

            data = {
                "sentence_good": sentence_good,
                "sentence_bad": sentence_bad,
                "two_prefix_prefix_good": sentence_good.split(" ", 2)[0] + " " + sentence_good.split(" ", 2)[1],
                "two_prefix_prefix_bad": sentence_bad.split(" ", 2)[0] + " " + sentence_bad.split(" ", 2)[1],
                "two_prefix_word": fixed_word.split(" ")[0],
            }
            return data, data["sentence_good"]

        raise LexicalGapError("No regular_plural_subject_verb_agreement_2 sample found after bounded retries")


def build_generator():
    return AgreementGenerator()


def main():
    generator = build_generator()
    generator.generate_paradigm(rel_output_path="outputs/blimp/%s.jsonl" % generator.uid)


if __name__ == "__main__":
    main()
