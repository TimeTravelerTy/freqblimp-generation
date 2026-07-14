# Critical-verb frequency factorial (2026-07-14)

This experiment separates the Zipf regime of the lexical contrast verb from
the Zipf regime of the remaining open-class material in selected FreqBLiMP
paradigms.

| Cell | Critical contrast verb | Context content words |
| --- | --- | --- |
| `verb_head_context_head` | head (3.5--5.5) | head (3.5--5.5) |
| `verb_xtail_context_head` | xtail (1.2--2.2) | head (3.5--5.5) |
| `verb_head_context_xtail` | head (3.5--5.5) | xtail (1.2--2.2) |
| `verb_xtail_context_xtail` | xtail (1.2--2.2) | xtail (1.2--2.2) |

The licensing set is `causative`, `inchoative`, `passive_1`, `passive_2`, and
`drop_argument`. The agreement controls are
`regular_plural_subject_verb_agreement_1` and
`determiner_noun_agreement_1`.

Every cell is regenerated with the current generator and seed rather than
reusing the published head/xtail files. The two object-raising paradigms are
excluded: their curated grammatical raising-predicate pool has no xtail
members, so it cannot support a critical-verb manipulation. A realised-slot
audit accompanies every generated factorial cell. For the selected licensing
paradigms, every sampled lexical verb is a contrast verb, so the ordinary verb
window is the critical-verb factor.

The test estimates the verb main effect, context main effect, and their
interaction.  It does not make the rare-verb condition collocation-neutral:
rare target verbs can still form rare or unattested verb--argument
combinations with head-frequency nouns.

The generated cells are stored under
`data/freqblimp/critical_verb_factorial_20260714/`. Before evaluation, run
`audit_realized_critical_slots` for the five licensing UIDs, with a
`--regime-bound` entry that assigns the verb window to each factorial cell.
