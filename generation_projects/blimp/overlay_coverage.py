"""Final BLiMP overlay coverage buckets.

"Done" means we intentionally support overlay expansion of a content-word slot
for the paradigm. "Remaining" is explicit so the audit stays stable even when
empty. "Skip" covers paradigms where we are not adding overlay-targeted hooks.
"""

DONE_OVERLAY_CASES = (
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

SKIP_FOR_OVERLAY = (
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "complex_NP_island",
    "coordinate_structure_constraint_complex_left_branch",
    "coordinate_structure_constraint_object_extraction",
    "coordinate_structure_constraint_subject_extraction",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "determiner_noun_agreement_irregular_1",
    "determiner_noun_agreement_irregular_2",
    "determiner_noun_agreement_with_adjective_1",
    "determiner_noun_agreement_with_adj_2",
    "determiner_noun_agreement_with_adj_irregular_1",
    "determiner_noun_agreement_with_adj_irregular_2",
    "distractor_agreement_relative_clause",
    "distractor_agreement_relational_noun",
    "irregular_past_participle_adjectives",
    "irregular_past_participle_verbs",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "left_branch_island_echo_question",
    "left_branch_island_simple_question",
    "matrix_question_npi_licensor_present",
    "npi_present_1",
    "npi_present_2",
    "only_npi_licensor_present",
    "only_npi_scope",
    "principle_A_c_command",
    "principle_A_case_1",
    "principle_A_case_2",
    "principle_A_domain_1",
    "principle_A_domain_2",
    "principle_A_domain_3",
    "principle_A_reconstruction",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope",
    "sentential_subject_island",
    "wh_island",
    "wh_questions_object_gap",
    "wh_questions_object_gap_long_distance",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "wh_vs_that_no_gap",
    "wh_vs_that_no_gap_long_distance",
    "wh_vs_that_with_gap",
    "wh_vs_that_with_gap_long_distance",
)
