"""Audit realised Zipf values at known critical slots in released FreqBLiMP data.

The released JSONL files do not retain generator choice traces.  This tool
therefore recovers the contrast predicate from the paired sentence fields and
scores its realised surface form with the same ``wordfreq`` source used by the
generator.  It is intentionally streaming and never reads overlay artifacts.
"""

import argparse
import csv
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median

from utils.frequency import zipf_for_expression


DEFAULT_UIDS = (
    "causative",
    "inchoative",
    "passive_1",
    "passive_2",
    "drop_argument",
    "existential_there_object_raising",
    "expletive_it_object_raising",
)

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _tokens(text):
    return TOKEN_RE.findall(str(text).lower())


def _paired_difference(record):
    good = _tokens(record.get("sentence_good", ""))
    bad = _tokens(record.get("sentence_bad", ""))
    differing = [
        (good[i1:i2], bad[j1:j2])
        for tag, i1, i2, j1, j2 in SequenceMatcher(a=good, b=bad).get_opcodes()
        if tag != "equal"
    ]
    if len(differing) != 1:
        raise ValueError("expected one differing span, found %d: %r" % (len(differing), differing))
    good_words, bad_words = differing[0]
    if not good_words or not bad_words:
        raise ValueError("expected a lexical predicate on both sides, found %r" % (differing[0],))
    # A BLiMP vocabulary expression can be a phrasal verb (e.g. ``wake up``
    # or ``baby sat``).  Preserve the whole replaced expression because that
    # is the string on which the generator applies its Zipf constraint.
    return " ".join(good_words), " ".join(bad_words)


def _object_raising_difference(uid, record):
    if uid == "existential_there_object_raising":
        good_prefix = str(record.get("two_prefix_prefix_good", "")).strip()
        bad_prefix = str(record.get("two_prefix_prefix_bad", "")).strip()
        good_words, bad_words = _tokens(good_prefix), _tokens(bad_prefix)
        if not good_words or not bad_words:
            raise ValueError("missing two-prefix fields")
        return good_words[-1], bad_words[-1]

    marker = " it to be"
    prefixes = (
        str(record.get("two_prefix_prefix_good", "")).strip().lower(),
        str(record.get("two_prefix_prefix_bad", "")).strip().lower(),
    )
    targets = []
    for prefix in prefixes:
        if marker not in prefix:
            raise ValueError("missing expletive-it marker in %r" % prefix)
        before_marker = prefix.split(marker, 1)[0]
        words = _tokens(before_marker)
        if not words:
            raise ValueError("missing matrix verb before expletive-it marker")
        targets.append(words[-1])
    return tuple(targets)


def _critical_words(uid, record):
    if uid in {"existential_there_object_raising", "expletive_it_object_raising"}:
        return _object_raising_difference(uid, record)
    return _paired_difference(record)


def _quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values, lower, upper):
    if not values:
        return {
            "n": 0,
            "in_window_n": 0,
            "in_window_pct": None,
            "below_window_n": 0,
            "above_window_n": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }
    in_window = [value for value in values if lower <= value <= upper]
    return {
        "n": len(values),
        "in_window_n": len(in_window),
        "in_window_pct": 100.0 * len(in_window) / len(values),
        "below_window_n": sum(value < lower for value in values),
        "above_window_n": sum(value > upper for value in values),
        "min": min(values),
        "p25": _quantile(values, 0.25),
        "median": median(values),
        "p75": _quantile(values, 0.75),
        "max": max(values),
    }


def _regime_bounds(regime, overrides=None):
    if overrides and regime in overrides:
        return overrides[regime]
    if regime == "head":
        return 3.5, 5.5
    if regime == "tail":
        return 2.4, 3.2
    if regime == "xtail":
        return 1.2, 2.2
    raise ValueError("unknown regime %r" % regime)


def _parse_regime_bounds(specifications):
    parsed = {}
    for specification in specifications or ():
        try:
            name, bounds = specification.split("=", 1)
            lower, upper = (float(value) for value in bounds.split(",", 1))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--regime-bound must be NAME=LOWER,UPPER; got %r" % specification
            ) from exc
        if not name:
            raise argparse.ArgumentTypeError("--regime-bound name must not be empty")
        parsed[name] = (lower, upper)
    return parsed


def _read_records(uid, regime, data_root):
    path = data_root / regime / (uid + ".jsonl")
    if not path.exists():
        return path, []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                good_word, bad_word = _critical_words(uid, record)
            except ValueError as exc:
                rows.append({
                    "uid": uid,
                    "regime": regime,
                    "line_number": line_number,
                    "side": "error",
                    "critical_word": "",
                    "zipf": "",
                    "error": str(exc),
                })
                continue
            for side, word in (("good", good_word), ("bad", bad_word)):
                rows.append({
                    "uid": uid,
                    "regime": regime,
                    "line_number": line_number,
                    "side": side,
                    "critical_word": word,
                    "zipf": zipf_for_expression(word),
                    "error": "",
                })
    return path, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/freqblimp")
    parser.add_argument("--regimes", nargs="+", default=("head", "tail", "xtail"))
    parser.add_argument("--uids", nargs="+", default=DEFAULT_UIDS)
    parser.add_argument(
        "--regime-bound",
        action="append",
        default=[],
        help="Override a regime window, e.g. verb_xtail_context_head=1.2,2.2. May be repeated.",
    )
    parser.add_argument("--output-dir", default="outputs/critical_slot_audit")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    regime_bounds = _parse_regime_bounds(args.regime_bound)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    missing = []
    for uid in args.uids:
        for regime in args.regimes:
            path, rows = _read_records(uid, regime, data_root)
            if not rows and not path.exists():
                missing.append(str(path))
                continue
            all_rows.extend(rows)

    summary_rows = []
    for uid in args.uids:
        for regime in args.regimes:
            lower, upper = _regime_bounds(regime, regime_bounds)
            for side in ("good", "bad"):
                values = [
                    float(row["zipf"])
                    for row in all_rows
                    if row["uid"] == uid and row["regime"] == regime and row["side"] == side and row["error"] == ""
                ]
                counts = Counter(
                    row["critical_word"]
                    for row in all_rows
                    if row["uid"] == uid and row["regime"] == regime and row["side"] == side and row["error"] == ""
                )
                row = {"uid": uid, "regime": regime, "side": side, "zipf_min_requested": lower, "zipf_max_requested": upper}
                row.update(_summary(values, lower, upper))
                row["unique_critical_words"] = len(counts)
                row["top_critical_words"] = "; ".join("%s:%d" % item for item in counts.most_common(8))
                summary_rows.append(row)

    errors = [row for row in all_rows if row["error"]]
    detail_path = output_dir / "critical_slot_detail.csv"
    summary_path = output_dir / "critical_slot_summary.csv"
    metadata_path = output_dir / "critical_slot_audit.json"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("uid", "regime", "line_number", "side", "critical_word", "zipf", "error"))
        writer.writeheader()
        writer.writerows(all_rows)
    fields = tuple(summary_rows[0]) if summary_rows else ()
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump({
            "data_root": str(data_root),
            "uids": list(args.uids),
            "regimes": list(args.regimes),
            "regime_bounds": regime_bounds,
            "missing_paths": missing,
            "extraction_error_count": len(errors),
            "summary": summary_rows,
        }, handle, indent=2, sort_keys=True)

    for row in summary_rows:
        print(
            "{uid:42s} {regime:5s} {side:4s} n={n:4d} in-window={in_window_n:4d} ({in_window_pct:5.1f}%) "
            "median={median!s:>5} range=[{min!s:>5}, {max!s:>5}]".format(**row)
        )
    if errors:
        print("[warn] %d extraction errors; inspect %s" % (len(errors), detail_path))
    if missing:
        print("[warn] %d missing dataset files" % len(missing))
    print("[saved] %s" % output_dir)


if __name__ == "__main__":
    main()
