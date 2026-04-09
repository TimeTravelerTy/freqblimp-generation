"""Final BLiMP overlay coverage buckets.

"Done" means we intentionally support overlay expansion of a content-word slot
for the paradigm. "Remaining" is explicit so the audit stays stable even when
empty. "Skip" covers paradigms where we are not adding overlay-targeted hooks.
"""

DONE_OVERLAY_CASES = (
    # Group A — varying slot wrapped
    ("passive_1", ("verb",)),
    ("passive_2", ("verb",)),
    ("drop_argument", ("verb",)),
    ("causative", ("verb",)),
    ("inchoative", ("verb",)),
    ("animate_subject_passive", ("noun",)),
    ("animate_subject_trans", ("noun",)),
    ("existential_there_subject_raising", ("adjective",)),
    ("existential_there_object_raising", ("verb",)),
    ("expletive_it_object_raising", ("verb",)),
    ("tough_vs_raising_1", ("adjective",)),
    ("tough_vs_raising_2", ("adjective",)),
    ("existential_there_quantifiers_1", ("noun",)),
    ("existential_there_quantifiers_2", ("noun",)),
    ("superlative_quantifiers_1", ("noun",)),
    ("superlative_quantifiers_2", ("noun",)),
    ("ellipsis_n_bar_1", ("noun", "adjective")),
    ("ellipsis_n_bar_2", ("noun", "adjective")),
    ("transitive", ("verb",)),
    ("intransitive", ("verb",)),
    # Group B1 — irregular paradigms: irregular slot closed, filler slots wrapped
    ("irregular_past_participle_verbs", ("noun",)),
    ("irregular_past_participle_adjectives", ("noun", "adjective")),
    ("irregular_plural_subject_verb_agreement_1", ("verb",)),
    ("irregular_plural_subject_verb_agreement_2", ("verb", "noun")),
    ("determiner_noun_agreement_irregular_1", ("verb", "noun")),
    ("determiner_noun_agreement_irregular_2", ("verb", "noun")),
    ("determiner_noun_agreement_with_adj_irregular_1", ("verb", "noun", "adjective")),
    ("determiner_noun_agreement_with_adj_irregular_2", ("verb", "noun", "adjective")),
    # Group B2 — reflexive/binding: refl_preds closed, embedding verbs and nouns wrapped
    ("anaphor_gender_agreement", ("noun",)),
    ("anaphor_number_agreement", ("noun",)),
    ("principle_A_domain_1", ("verb", "noun")),
    ("principle_A_domain_2", ("verb", "noun")),
    ("principle_A_domain_3", ("verb", "noun")),
    ("principle_A_reconstruction", ("noun",)),
    ("principle_A_c_command", ("verb", "noun")),
    ("principle_A_case_1", ("verb", "noun")),
    ("principle_A_case_2", ("verb", "noun")),
    # Group B3 — syntactic island tests: gap position varies, verbs + nouns are fillers
    ("adjunct_island", ("verb", "noun")),
    ("complex_NP_island", ("noun",)),
    ("wh_island", ("verb", "noun")),
    ("left_branch_island_simple_question", ("verb", "noun")),
    ("left_branch_island_echo_question", ("verb", "noun")),
    ("coordinate_structure_constraint_subject_extraction", ("verb", "noun")),
    ("coordinate_structure_constraint_object_extraction", ("verb", "noun")),
    ("coordinate_structure_constraint_complex_left_branch", ("verb", "noun")),
    ("sentential_subject_island", ("verb", "noun")),
    # Group B4 — wh-question and that/wh complementizer tests
    ("wh_questions_object_gap", ("verb", "noun")),
    ("wh_questions_subject_gap", ("verb", "noun")),
    ("wh_questions_object_gap_long_distance", ("verb", "noun")),
    ("wh_questions_subject_gap_long_distance", ("verb", "noun")),
    ("wh_vs_that_no_gap", ("verb", "noun")),
    ("wh_vs_that_no_gap_long_distance", ("verb", "noun")),
    ("wh_vs_that_with_gap", ("verb", "noun")),
    ("wh_vs_that_with_gap_long_distance", ("verb", "noun")),
    # Group B5 — morphological agreement tests
    ("determiner_noun_agreement_1", ("verb", "noun")),
    ("determiner_noun_agreement_2", ("verb", "noun")),
    ("determiner_noun_agreement_with_adjective_1", ("verb", "noun", "adjective")),
    ("determiner_noun_agreement_with_adj_2", ("verb", "noun", "adjective")),
    ("distractor_agreement_relative_clause", ("verb", "noun")),
    ("distractor_agreement_relational_noun", ("verb", "noun")),
    ("regular_plural_subject_verb_agreement_1", ("verb", "noun")),
    ("regular_plural_subject_verb_agreement_2", ("verb", "noun")),
    # Group B6 — NPI tests: polarity trigger varies, verb is open-class filler
    ("npi_present_1", ("verb",)),
    ("npi_present_2", ("verb",)),
    ("matrix_question_npi_licensor_present", ("verb",)),
    ("only_npi_licensor_present", ("verb",)),
    ("only_npi_scope", ("verb", "noun")),
    ("sentential_negation_npi_licensor_present", ("verb",)),
    ("sentential_negation_npi_scope", ("verb", "noun")),
)

DONE_PARADIGMS = tuple(uid for uid, _controlled_pos in DONE_OVERLAY_CASES)

# These were the last content-word targets that still needed a fresh-process
# runtime check. They expand through overlay without any additional local hook.
RUNTIME_VERIFIED_WITHOUT_NEW_HOOKS = (
    "ellipsis_n_bar_1",
    "transitive",
    "intransitive",
)

REMAINING_CONTENT_WORD_TARGETS = ()

SKIP_FOR_OVERLAY = ()
