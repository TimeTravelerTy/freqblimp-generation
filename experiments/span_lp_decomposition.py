#!/usr/bin/env python3
"""Decompose paired-sentence LP margins into critical, licensing, and remainder spans.

This diagnostic is tailored to the three existential/raising paradigms whose
minimal pairs replace one predicate while preserving the rest of the sentence.
It deliberately scores the plain sentence only: for each pair the shared
prefix has the same token-level likelihood on both sides, so the full LP
margin is the sum of (i) the critical predicate, (ii) the expletive licensing
window, and (iii) the remaining suffix.  A per-item assertion checks that
identity rather than assuming it.

The script lives in FreqBLiMP because it consumes its released datasets, but
uses blimp-rare's ``LlamaNLLScorer`` for model loading.  On TSUBAME pass the
clean blimp-rare evaluation checkout with ``--blimp-rare-root``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Sequence


UID_CONFIG = {
    "existential_there_object_raising": {
        "critical_label": "matrix_verb",
        "licensing_window": " there to be",
    },
    "expletive_it_object_raising": {
        "critical_label": "matrix_verb",
        "licensing_window": " it to be",
    },
    "existential_there_subject_raising": {
        "critical_label": "predicate",
        "licensing_window": " to be",
    },
}
DEFAULT_UIDS = tuple(UID_CONFIG)
DEFAULT_REGIMES = ("head", "tail", "xtail")


@dataclass(frozen=True)
class SpanLayout:
    prefix_end: int
    critical_start: int
    critical_end: int
    licensing_end: int

    def category(self, char_start: int) -> str:
        """Assign a token by the first raw-sentence character it covers."""
        if char_start < self.critical_start:
            return "prefix"
        if char_start < self.critical_end:
            return "critical_locus"
        if char_start < self.licensing_end:
            return "licensing_window"
        return "remainder"


@dataclass(frozen=True)
class Item:
    uid: str
    regime: str
    pair_id: str
    good_text: str
    bad_text: str
    good_layout: SpanLayout
    bad_layout: SpanLayout
    good_lemma: str
    bad_lemma: str


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _layout_pair(uid: str, good: str, bad: str) -> tuple[SpanLayout, SpanLayout, str, str]:
    """Recover the one replaced predicate and fixed licensing window by offsets."""
    try:
        marker = UID_CONFIG[uid]["licensing_window"]
    except KeyError as exc:
        raise ValueError(f"Unsupported UID: {uid}") from exc
    prefix_end = _common_prefix_length(good, bad)
    # Do not recover a span by the longest common *character* suffix: words
    # such as ``discovered``/``advised`` share ``ed``, which would wrongly
    # peel a morphological fragment off the predicate.  The fixed marker gives
    # the exact lexical right boundary instead.
    good_critical_end = good.find(marker, prefix_end)
    bad_critical_end = bad.find(marker, prefix_end)
    if prefix_end == 0 or good_critical_end < 0 or bad_critical_end < 0 or prefix_end >= good_critical_end or prefix_end >= bad_critical_end:
        raise ValueError(f"{uid}: pair is not one nonempty replacement: {good!r} / {bad!r}")
    # The lexical predicate begins exactly where the shared prefix ends.  Its
    # end must be the first character of the prescribed continuation.
    if not good[good_critical_end:].startswith(marker):
        raise ValueError(
            f"{uid}: good suffix does not begin with {marker!r}: {good[good_critical_end:]!r}"
        )
    if not bad[bad_critical_end:].startswith(marker):
        raise ValueError(
            f"{uid}: bad suffix does not begin with {marker!r}: {bad[bad_critical_end:]!r}"
        )
    if good[good_critical_end:] != bad[bad_critical_end:]:
        raise ValueError(f"{uid}: continuation is not string-identical after the critical predicate")
    good_layout = SpanLayout(prefix_end, prefix_end, good_critical_end, good_critical_end + len(marker))
    bad_layout = SpanLayout(prefix_end, prefix_end, bad_critical_end, bad_critical_end + len(marker))
    good_lemma = good[prefix_end:good_critical_end].strip().lower()
    bad_lemma = bad[prefix_end:bad_critical_end].strip().lower()
    if not good_lemma or not bad_lemma:
        raise ValueError(f"{uid}: empty critical predicate after layout recovery")
    return good_layout, bad_layout, good_lemma, bad_lemma


def _read_items(data_root: Path, uids: Sequence[str], regimes: Sequence[str], expected_pairs: int | None) -> list[Item]:
    items: list[Item] = []
    for regime in regimes:
        for uid in uids:
            path = data_root / regime / f"{uid}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    good = record.get("sentence_good")
                    bad = record.get("sentence_bad")
                    if not isinstance(good, str) or not isinstance(bad, str):
                        raise ValueError(f"{path}:{line_number}: missing sentence_good/sentence_bad")
                    good_layout, bad_layout, good_lemma, bad_lemma = _layout_pair(uid, good, bad)
                    pair_id = str(record.get("pairID", line_number - 1))
                    items.append(
                        Item(uid, regime, pair_id, good, bad, good_layout, bad_layout, good_lemma, bad_lemma)
                    )
                    count += 1
            if expected_pairs is not None and count != expected_pairs:
                raise ValueError(f"{path}: expected {expected_pairs} pairs, found {count}")
            print(f"[Data] {regime}/{uid}: {count} pairs")
    return items


def _batch_ranges(length: int, batch_size: int) -> Iterable[range]:
    for start in range(0, length, batch_size):
        yield range(start, min(start + batch_size, length))


def _score_text_batch(scorer, texts: Sequence[str], layouts: Sequence[SpanLayout], max_length: int) -> list[dict[str, float]]:
    """Return total LP and LP per first-character-assigned span for each text."""
    import torch

    tokenizer = scorer.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Span alignment needs a fast tokenizer with offset mappings.")
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    offsets = encoded.pop("offset_mapping")
    input_ids = encoded["input_ids"].to(scorer.device, non_blocking=True)
    attention_mask = encoded["attention_mask"].to(scorer.device, non_blocking=True)
    use_amp = scorer.device.type in {"cuda", "mps"}
    with torch.inference_mode():
        with torch.amp.autocast(device_type=scorer.device.type, dtype=scorer.dtype, enabled=use_amp):
            outputs = scorer.model(input_ids=input_ids, attention_mask=attention_mask)
    # Score in float32.  This makes the reconstruction check about span logic,
    # rather than bfloat16 accumulation order.
    token_logprobs = torch.log_softmax(outputs.logits.float()[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    token_logprobs = token_logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).cpu()
    shift_mask = attention_mask[:, 1:].cpu()
    offsets = offsets.cpu()
    results: list[dict[str, float]] = []
    for row_index, layout in enumerate(layouts):
        sums = {"prefix": 0.0, "critical_locus": 0.0, "licensing_window": 0.0, "remainder": 0.0}
        for position in range(token_logprobs.shape[1]):
            if not bool(shift_mask[row_index, position]):
                continue
            token_start, token_end = (int(value) for value in offsets[row_index, position + 1].tolist())
            if token_end <= token_start:
                raise RuntimeError("Encountered a scored token with no character offset.")
            sums[layout.category(token_start)] += float(token_logprobs[row_index, position])
        sums["total_lp"] = sum(sums.values())
        results.append(sums)
    return results


def _score_items(scorer, model: str, items: Sequence[Item], batch_size: int, max_length: int, max_residual: float) -> list[dict]:
    rows: list[dict] = []
    for batch_number, indices in enumerate(_batch_ranges(len(items), batch_size), start=1):
        batch_items = [items[index] for index in indices]
        texts: list[str] = []
        layouts: list[SpanLayout] = []
        for item in batch_items:
            texts.extend((item.good_text, item.bad_text))
            layouts.extend((item.good_layout, item.bad_layout))
        scores = _score_text_batch(scorer, texts, layouts, max_length=max_length)
        for item_index, item in enumerate(batch_items):
            good = scores[2 * item_index]
            bad = scores[2 * item_index + 1]
            delta_critical = good["critical_locus"] - bad["critical_locus"]
            delta_window = good["licensing_window"] - bad["licensing_window"]
            delta_remainder = good["remainder"] - bad["remainder"]
            delta_total = good["total_lp"] - bad["total_lp"]
            reconstructed = delta_critical + delta_window + delta_remainder
            residual = delta_total - reconstructed
            if abs(residual) > max_residual:
                raise AssertionError(
                    f"{item.uid}/{item.regime}/pairID={item.pair_id}: residual {residual:.8g} "
                    f"exceeds {max_residual}; prefix delta={good['prefix'] - bad['prefix']:.8g}"
                )
            if good["critical_locus"] == 0.0 or bad["critical_locus"] == 0.0:
                raise AssertionError(f"{item.uid}/{item.regime}/pairID={item.pair_id}: empty critical token span")
            rows.append(
                {
                    "model": model,
                    "uid": item.uid,
                    "regime": item.regime,
                    "pairID": item.pair_id,
                    "good_critical_lemma": item.good_lemma,
                    "bad_critical_lemma": item.bad_lemma,
                    "good_text": item.good_text,
                    "bad_text": item.bad_text,
                    "good_total_lp": good["total_lp"],
                    "bad_total_lp": bad["total_lp"],
                    "delta_lp": delta_total,
                    "delta_critical_locus_lp": delta_critical,
                    "delta_licensing_window_lp": delta_window,
                    "delta_remainder_lp": delta_remainder,
                    "prefix_delta_lp": good["prefix"] - bad["prefix"],
                    "reconstruction_residual_lp": residual,
                    "correctness": int(delta_total > 0.0),
                }
            )
        if batch_number % 25 == 0 or batch_number == math.ceil(len(items) / batch_size):
            print(f"[Score] {model}: {min(indices.stop, len(items))}/{len(items)} pairs")
    return rows


MEAN_COLUMNS = (
    "delta_lp",
    "delta_critical_locus_lp",
    "delta_licensing_window_lp",
    "delta_remainder_lp",
    "prefix_delta_lp",
    "reconstruction_residual_lp",
    "correctness",
)


def _mean_row(rows: Sequence[dict], extra: dict) -> dict:
    output = dict(extra)
    output["n_items"] = len(rows)
    for column in MEAN_COLUMNS:
        output[f"mean_{column}"] = fmean(float(row[column]) for row in rows)
    output["max_abs_reconstruction_residual_lp"] = max(abs(float(row["reconstruction_residual_lp"])) for row in rows)
    return output


def _group_means(rows: Sequence[dict], keys: Sequence[str]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [_mean_row(group, dict(zip(keys, key))) for key, group in sorted(groups.items())]


def _lemma_means(rows: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        for side, column in (("good", "good_critical_lemma"), ("bad", "bad_critical_lemma")):
            grouped[(row["model"], row["uid"], row["regime"], side, row[column])].append(row)
    lemma_rows = [
        _mean_row(group, {"model": key[0], "uid": key[1], "regime": key[2], "side": key[3], "critical_lemma": key[4]})
        for key, group in sorted(grouped.items())
    ]
    clustering: list[dict] = []
    cluster_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in lemma_rows:
        cluster_groups[(row["model"], row["uid"], row["regime"], row["side"])].append(row)
    for key, group in sorted(cluster_groups.items()):
        counts = [int(row["n_items"]) for row in group]
        lemma_margin_means = [float(row["mean_delta_lp"]) for row in group]
        total = sum(counts)
        clustering.append(
            {
                "model": key[0], "uid": key[1], "regime": key[2], "side": key[3],
                "n_lemmas": len(group), "n_items": total,
                "largest_lemma_n": max(counts), "largest_lemma_share": max(counts) / total,
                "between_lemma_sd_mean_delta_lp": pstdev(lemma_margin_means) if len(lemma_margin_means) > 1 else 0.0,
                "interpretation": "few-lemma diagnostic; do not treat item-level n as lexical generalization",
            }
        )
    return lemma_rows, clustering


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, rows: Sequence[dict], model: str, uids: Sequence[str], regimes: Sequence[str]) -> None:
    by_uid_regime = _group_means(rows, ("model", "uid", "regime"))
    lines = [
        "# Span-wise LP decomposition",
        "",
        f"Model: `{model}`. Regimes: {', '.join(regimes)}. Paradigms: {', '.join(uids)}.",
        "",
        "Each pair is partitioned by raw character offsets into the replaced critical predicate, the fixed expletive licensing window, and the remainder. Tokens are assigned by their first character offset. The shared-prefix contribution is asserted to be zero up to the configured tolerance, so the three reported terms reconstruct the full sentence LP margin.",
        "",
        "| UID | Regime | n | Δ total LP | Δ critical locus | Δ licensing window | Δ remainder | accuracy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in by_uid_regime:
        lines.append(
            "| {uid} | {regime} | {n_items} | {mean_delta_lp:+.3f} | {mean_delta_critical_locus_lp:+.3f} | {mean_delta_licensing_window_lp:+.3f} | {mean_delta_remainder_lp:+.3f} | {mean_correctness:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "Lemma-level files retain the contrast margins while grouping independently by the good-side and bad-side curated predicate. They are descriptive diagnostics, not evidence for a population-level raising/control contrast.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--blimp-rare-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--uids", nargs="+", default=list(DEFAULT_UIDS), choices=list(DEFAULT_UIDS))
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES), choices=list(DEFAULT_REGIMES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--expected-pairs", type=int, default=1000)
    parser.add_argument("--max-residual", type=float, default=1e-4)
    parser.add_argument("--dry-run", action="store_true", help="Validate layouts/data without loading a model.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    items = _read_items(args.data_root, args.uids, args.regimes, args.expected_pairs)
    if args.dry_run:
        print(f"[Dry run] validated {len(items)} pairs with nonempty critical spans.")
        return
    sys.path.insert(0, str(args.blimp_rare_root))
    import torch
    from src.sentence_nll import LlamaNLLScorer

    for model in args.models:
        print(f"[Model] Loading {model}")
        scorer = LlamaNLLScorer(
            model_name=model,
            dtype=args.dtype,
            device_map=args.device_map,
            padding_side="left",
        )
        rows = _score_items(
            scorer, model, items, batch_size=args.batch_size,
            max_length=args.max_length, max_residual=args.max_residual,
        )
        slug = model.split("/")[-1].replace(".", "_")
        model_dir = args.out_dir / slug
        _write_csv(model_dir / "per_item.csv", rows)
        _write_csv(model_dir / "summary_by_uid_regime.csv", _group_means(rows, ("model", "uid", "regime")))
        _write_csv(model_dir / "summary_by_regime.csv", _group_means(rows, ("model", "regime")))
        lemma_rows, clustering = _lemma_means(rows)
        _write_csv(model_dir / "by_critical_lemma.csv", lemma_rows)
        _write_csv(model_dir / "lemma_clustering.csv", clustering)
        _write_readme(model_dir / "README.md", rows, model, args.uids, args.regimes)
        print(f"[Saved] {model_dir}")
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
