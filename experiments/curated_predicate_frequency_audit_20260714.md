# Curated predicate frequency audit (2026-07-14)

The released head, tail, and xtail data were audited at the contrast predicate
itself, rather than by aggregating all controlled content-word slots. The audit
recovers the changed predicate from each minimal pair and applies the same
`wordfreq` Zipf score used by the generator. It covers all 1,000 items per
side and regime; multiword predicate expressions are scored as expressions.

| Paradigm | Tail good / bad in window | Xtail good / bad in window | Consequence |
| --- | ---: | ---: | --- |
| causative | 100% / 100% | 100% / 100% | clean contrast-verb manipulation |
| inchoative | 100% / 100% | 100% / 100% | clean contrast-verb manipulation |
| passive_1 | 100% / 100% | 100% / 100% | clean contrast-verb manipulation |
| passive_2 | 100% / 100% | 100% / 100% | clean contrast-verb manipulation |
| drop_argument | 100% / 100% | 100% / 100% | clean contrast-verb manipulation |
| existential-there object raising | 11.9% / 100% | 0% / 31.1% | critical matrix-verb manipulation unavailable |
| expletive-it object raising | 12.6% / 100% | 0% / 32.6% | critical matrix-verb manipulation unavailable |
| existential-there subject raising | 100% / 100% | 0% / 74.1% | curated predicate pool relaxed in xtail |
| tough vs. raising 1 | 100% / 10.7% | 10.5% / 0% | curated adjective pools relaxed |
| tough vs. raising 2 | 11.2% / 100% | 0% / 10.3% | curated adjective pools relaxed |

The object-raising grammatical matrix predicates in xtail have median Zipf
3.62 (range 2.99--3.73); none is genuinely xtail. The same phenomenon affects
other curated raising/tough predicate pools, so those paradigms require an
explicit per-slot caveat in the paper rather than being treated as evidence for
a rare critical predicate.

The source is `_curated_rows_for_expression_families` in
`generation_projects/blimp/overlay_guards.py`. Its historical minimum-candidate
rule padded depleted curated pools with the nearest available frequencies. It
now emits a conspicuous runtime warning whenever this happens. The new
`--strict-curated-zipf` mode rejects that padding; it should be used for any
future experiment that claims a curated predicate is frequency-controlled.

The critical-verb factorial therefore excludes the two object-raising
paradigms and uses the five clean licensing paradigms reported above. The full
audit is reproducible with
`python -m generation_projects.blimp.audit_realized_critical_slots`.
