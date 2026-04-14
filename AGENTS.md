# Memory Safety

- Treat overlay mode as memory-sensitive by default. `vocabulary_overlay.csv`, `vocabulary_overlay_manifest.json`, and `outputs/cache/vocabulary_frequency_cache.json` are large enough to cause OOMs if loaded eagerly.
- Do not `json.load()` `vocabulary_overlay_manifest.json` or `outputs/cache/vocabulary_frequency_cache.json` in ad hoc scripts. Inspect them with file-size checks, line counts, or narrowly scoped streaming tools instead.
- Do not run broad overlay generation or profiling jobs casually. Prefer one UID at a time, `python3 -m generation_projects.blimp.run ... --no-trace --no-progress`, and avoid `all`.
- Before adding new overlay-facing generator logic, avoid patterns that materialize and cache large per-candidate arrays inside scans. Prefer capped scans, label-index checks, and reusable cached indices.
- If memory behavior must be investigated, start with source inspection and artifact sizes before running Python that imports overlay assets.
