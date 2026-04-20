# Memory Safety

- Treat overlay mode as memory-sensitive by default. `vocabulary_overlay.csv`, `vocabulary_overlay_manifest.json`, and `outputs/cache/vocabulary_frequency_cache.json` are large enough to cause OOMs if loaded eagerly.
- Do not `json.load()` `vocabulary_overlay_manifest.json` or `outputs/cache/vocabulary_frequency_cache.json` in ad hoc scripts. Inspect them with file-size checks, line counts, or narrowly scoped streaming tools instead.
- Do not run broad overlay generation or profiling jobs casually. Prefer one UID at a time, `python3 -m generation_projects.blimp.run ... --no-trace --no-progress`, and avoid `all`.
- Before adding new overlay-facing generator logic, avoid patterns that materialize and cache large per-candidate arrays inside scans. Prefer capped scans, label-index checks, and reusable cached indices.
- If memory behavior must be investigated, start with source inspection and artifact sizes before running Python that imports overlay assets.

# Session Handoff (2026-04-20, continued)

## What Was Committed This Session (5db47fe)

Single commit "Zipf-filter all constrainable slots + perf fixes for full 3-regime regen":

### Performance fixes
- `utils/vocab_table.py`: `clear_query_caches()` no longer clears `_EXPRESSION_ZIPF_REGISTRY`; added `register_query_cache_clear_hook` so external modules can register cache-clear callbacks.
- `generation_projects/blimp/overlay_guards.py`: `filter_rows_for_active_zipf` is now memoized (`_ZIPF_FILTER_CACHE` keyed on table identity + POS + Zipf bounds). Cache cleared via hook.
- `utils/data_generator.py`: Added `FREQBLIMP_PARADIGM_TIMEOUT_S` (per-UID wall-clock cap) and `FREQBLIMP_VERBOSE_FAILURE_LOG` (gate expensive log writes for expected exceptions).

### Correctness fixes
- `causative.py`, `inchoative.py`: Subj/Obj now Zipf-filtered with `fallback_on_empty=False`.
- `passive_1.py`, `passive_2.py`: bad verb pool changed from `passive=0` (too broad) to `strict_intrans=1 AND passive=0` (purely intransitive, no transitive frame). NP1/NP2 Zipf-filtered. Bounded 512-retry added.
  - **Key annotation note**: `strict_intrans=1` verbs have `strict_trans=""` (not `"0"`) in the vocab. Do NOT filter `strict_trans="0"`.
- `drop_argument.py`: Subj Zipf-filtered; bounded retry added.
- `animate_subject_transitive.py`, `animate_subject_passive.py`: verb slot + all noun slots Zipf-filtered; bounded retry added.
- `existential_there_subject_raising.py`: embedded subject + -ing verb Zipf-filtered (`fallback_on_empty=False`); curated predicate rows filtered by Zipf with `fallback_on_empty=True`.

### New files
- `generation_projects/blimp/feasibility_probe.py`: quick per-UID × regime acceptance-rate probe (20 attempts, 15s timeout per UID). Run before full regen to catch stall-risk UIDs.
- `generation_projects/blimp/overlay_coverage_report.py`: xtail source-vs-overlay lemma counts + argument-structure feasibility.

## Feasibility Probe Results (xtail, rebuilt overlay outputs/overlay_rebuild_20260420/)

- passive_1, passive_2, drop_argument, animate_subject_passive: 100% acceptance ✓
- causative, inchoative: ~3s/sample at xtail (slow but ~50 min/paradigm, within 12h limit)
- existential_there_subject_raising: 30% acceptance (~55 min/paradigm at xtail)
- animate_subject_transitive: init_failed in probe (memory pressure artifact from preceding long runs); direct test confirms it works correctly

## Rebuilt Overlay (outputs/overlay_rebuild_20260420/)

- noun lemmas: 10073 (vs old 10149, -76 negligible regression)
- verb lemmas: 390 (vs old 252, +138 significant improvement)
- adjective lemmas: 2834 (vs old 2754, +80)
- Decision: **promoted to TSUBAME canonical** (rsync in progress as of this handoff)
- After rsync: delete `overlay.csv.runtime.compact.v2.npy` on TSUBAME so it rebuilds from new overlay.csv

## TSUBAME State

- Job scripts uploaded: `jobs/fbr_head.sh`, `jobs/fbr_tail.sh`, `jobs/fbr_xtail.sh`
  - node_q=1 (48 cores), h_rt=12:00:00, --jobs 8
  - Output dirs: `outputs/tsubame_runs_refresh_20260421/{head,tail,xtail}/`
  - Env: FREQBLIMP_MAX_FAILURES=5000, FREQBLIMP_PARADIGM_TIMEOUT_S=900
  - Zipf: head=3.5-5.5, tail=2.4-3.2, xtail=1.2-2.2
- **NOT YET SUBMITTED** — waiting for overlay rsync to complete
- Previous runs (Apr 18-20) remain in `outputs/blimp-rare/` on TSUBAME storage

## Recommended Next Steps

1. Confirm rsync of rebuilt overlay to TSUBAME `outputs/overlay_full/` completed
2. Delete old .npy compact cache: `ssh tsubame "rm -f .../overlay_full/overlay.csv.runtime.compact.v2.npy"`
3. Submit 3 independent jobs: `qsub -g tga-sip_arase jobs/fbr_{head,tail,xtail}.sh` from `$PROJ`
4. Monitor with `qstat`; check logs under `logs/`
5. After jobs complete: score repaired paradigms (causative, inchoative, passive_1, passive_2, drop_argument, animate_subject_trans, animate_subject_passive, existential_there_subject_raising) for all 3 regimes × 3 models
6. If those UIDs improve ≥5 pts on ≥2 models vs Apr-20 results → promote full dataset and run full rescore

## Accuracy Baseline (Apr-20 runs, for comparison)

- Head: gemma-4-E4B lp=0.7039, Llama-3.1-8B lp=0.7167, Mistral-7B lp=0.7264
- Tail: gemma-4-E4B lp=0.6997, Llama-3.1-8B lp=0.7106, Mistral-7B lp=0.7340
