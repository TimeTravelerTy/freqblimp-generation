# FreqBLiMP

FreqBLiMP is a frequency-controlled BLiMP-style minimal-pair dataset and
generation pipeline. This repository contains the modified BLiMP generator, the
base vocabulary, and the final head/tail/xtail datasets used in the paper.

Evaluation, scoring, and paper-analysis scripts live in the companion
[`freq-blimp-eval`](https://github.com/TimeTravelerTy/blimp-rare) repository.

## What Is Included

- `data/freqblimp/{head,tail,xtail}/`: final paper datasets, 67 paradigms per
  regime and 1,000 minimal pairs per paradigm.
- `paper_scope_uids.txt`: the 67 BLiMP-paper-scope paradigms used by default.
- `generation_projects/blimp/`: modified BLiMP generators, frequency overlay
  guards, registry, and runner.
- `utils/`: shared vocabulary querying, sampling, morphology, and generation
  utilities.
- `vocabulary.csv` and `vocab_documentation.md`: base vocabulary and feature
  documentation.

The large generated overlay files are intentionally not tracked in Git:
`vocabulary_overlay.csv`, `vocabulary_overlay_manifest.json`, and
`outputs/cache/vocabulary_frequency_cache.json`. Use the release/artifact bundle
when you need to regenerate datasets with the expanded vocabulary.

## Dataset Layout

Each regime directory contains one `.jsonl` file and one `.manifest.json` file
per paradigm:

```text
data/freqblimp/
  head/
  tail/
  xtail/
```

The regimes correspond to Zipf frequency windows:

- `head`: 3.5-5.5
- `tail`: 2.4-3.2
- `xtail`: 1.2-2.2

The dataset is the 67-subtask paper-scope set. The extra generator files
`coordinate_structure_constraint_subject_extraction` and
`wh_questions_object_gap_long_distance` are retained for compatibility with this
repo's generator history, but they are not part of the default paper-scope
dataset.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For generation with the expanded vocabulary, download the overlay artifact and
place these files at the repo root:

```text
vocabulary_overlay.csv
vocabulary_overlay_manifest.json
outputs/cache/vocabulary_frequency_cache.json
```

Memory note: do not `json.load()` the overlay manifest or frequency cache in
ad hoc scripts. They are large enough to cause avoidable memory pressure. Inspect
them with file sizes, line counts, or narrowly scoped streaming reads.

## Generate Data

Generate one paradigm without the overlay:

```bash
python3 -m generation_projects.blimp.run transitive \
  --number-to-generate 100 \
  --output-dir outputs/local_smoke/transitive \
  --no-trace \
  --no-progress
```

Generate the paper-scope regimes with the overlay:

```bash
UIDS="$(tr '\n' ' ' < paper_scope_uids.txt)"

python3 -m generation_projects.blimp.run $UIDS \
  --number-to-generate 1000 \
  --seed 0 \
  --controlled-pos noun verb adjective \
  --zipf-min-all 3.5 \
  --zipf-max-all 5.5 \
  --use-overlay \
  --overlay-path vocabulary_overlay.csv \
  --overlay-manifest-path vocabulary_overlay_manifest.json \
  --frequency-cache-path outputs/cache/vocabulary_frequency_cache.json \
  --output-dir outputs/generated/head \
  --jobs 1 \
  --no-trace \
  --no-progress
```

Change the Zipf bounds and output directory for `tail` and `xtail`.

## Development Notes

- Generated outputs, caches, logs, and overlay files are ignored by Git.
- The tracked `data/freqblimp` directory is the curated public dataset, not a
  scratch output directory.
- `generation_projects/blimp/tools/` contains generator maintenance helpers such
  as feasibility and overlay coverage probes.
- Keep overlay-facing logic memory-conscious: prefer streaming, capped scans,
  label-index checks, and cached indices over eager materialization.

## Citation

If you use the generator lineage, cite BLiMP:

```bibtex
@article{warstadt2019blimp,
  title={BLiMP: A Benchmark of Linguistic Minimal Pairs for English},
  author={Warstadt, Alex and Parrish, Alicia and Liu, Haokun and Mohananey, Anhad and Peng, Wei and Wang, Sheng-Fu and Bowman, Samuel R.},
  journal={arXiv preprint arXiv:1912.00582},
  year={2019}
}
```
