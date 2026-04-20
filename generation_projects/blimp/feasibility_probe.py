"""
Quick feasibility probe: for each UID x regime, attempt N sample() calls and report
acceptance rate, timing, and stall risk. Run this before full regen to catch empty
Zipf pools and slow-convergence UIDs early.

Usage (with overlay):
    FREQBLIMP_USE_OVERLAY=1 \
    FREQBLIMP_VOCAB_OVERLAY=<path>/overlay.csv \
    FREQBLIMP_OVERLAY_MANIFEST=<path>/manifest.json \
    FREQBLIMP_FREQUENCY_CACHE=<path>/frequency_cache.json \
    python3 -m generation_projects.blimp.feasibility_probe \
        --overlay-path <path>/overlay.csv \
        --overlay-manifest-path <path>/manifest.json \
        --frequency-cache-path <path>/frequency_cache.json \
        --output feasibility.csv

Without overlay (base vocab only):
    python3 -m generation_projects.blimp.feasibility_probe --output feasibility.csv
"""

import argparse
import csv
import importlib
import os
import signal
import sys
import time

# Env vars must be set BEFORE any vocab_table / constituent_building imports.
# We parse args early and apply them before any other imports.

def _parse_args():
    parser = argparse.ArgumentParser(description="Feasibility probe: quick per-UID x regime sampling test")
    parser.add_argument("--overlay-path", default=None)
    parser.add_argument("--overlay-manifest-path", default=None)
    parser.add_argument("--frequency-cache-path", default=None)
    parser.add_argument("--uids", nargs="*", default=None, help="UIDs to probe (default: all)")
    parser.add_argument("--regimes", nargs="*", default=["head", "tail", "xtail"],
                        choices=["head", "tail", "xtail"])
    parser.add_argument("--attempts", type=int, default=20, help="Sample attempts per UID x regime")
    parser.add_argument("--uid-timeout", type=float, default=15.0, help="Wall-clock timeout per UID x regime (seconds)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: stdout)")
    parser.add_argument("--stall-threshold", type=float, default=0.1,
                        help="Acceptance rate below this is flagged as stall_risk")
    return parser.parse_args()


_REGIME_BOUNDS = {
    "head": (3.5, 5.5),
    "tail": (2.4, 3.2),
    "xtail": (1.2, 2.2),
}

_IGNORED_UIDS = {"__init__", "run", "build_overlay", "sbatch_generator",
                 "overlay_guards", "overlay_coverage_report", "feasibility_probe"}


def _available_uids():
    from pathlib import Path
    root = Path(__file__).resolve().parent
    return sorted(
        p.stem for p in root.glob("*.py")
        if p.stem not in _IGNORED_UIDS
    )


def _apply_overlay_env(args):
    if args.overlay_path:
        os.environ["FREQBLIMP_USE_OVERLAY"] = "1"
        os.environ["FREQBLIMP_VOCAB_OVERLAY"] = args.overlay_path
    if args.overlay_manifest_path:
        os.environ["FREQBLIMP_OVERLAY_MANIFEST"] = args.overlay_manifest_path
    if args.frequency_cache_path:
        os.environ["FREQBLIMP_FREQUENCY_CACHE"] = args.frequency_cache_path
    os.environ["FREQBLIMP_SHOW_PROGRESS"] = "0"
    os.environ["FREQBLIMP_VERBOSE_FAILURE_LOG"] = "0"


def _build_generator(uid):
    """Import UID module and instantiate its generator."""
    module = importlib.import_module("generation_projects.blimp.%s" % uid)
    if hasattr(module, "build_generator"):
        return module.build_generator()
    raise RuntimeError("No build_generator() in %s" % uid)


def _probe_uid_regime(uid, regime, attempts, timeout_s):
    """
    Attempt `attempts` sample() calls under a given regime policy.
    Returns a dict with accepted, failed, elapsed_s.
    """
    from utils.randomize import SamplingPolicy, configure_sampling_policy, clear_sampling_policy
    from utils.vocab_table import clear_query_caches
    from utils.exceptions import FrequencyConstraintError, LexicalGapError

    lo, hi = _REGIME_BOUNDS[regime]
    policy = SamplingPolicy(
        seed=42,
        controlled_pos=("noun", "verb", "adjective"),
        zipf_min={"noun": lo, "verb": lo, "adjective": lo},
        zipf_max={"noun": hi, "verb": hi, "adjective": hi},
        overlay_enabled=os.environ.get("FREQBLIMP_USE_OVERLAY") == "1",
        overlay_path=os.environ.get("FREQBLIMP_VOCAB_OVERLAY"),
        overlay_manifest_path=os.environ.get("FREQBLIMP_OVERLAY_MANIFEST"),
        frequency_cache_path=os.environ.get("FREQBLIMP_FREQUENCY_CACHE"),
    )
    configure_sampling_policy(policy)
    try:
        try:
            generator = _build_generator(uid)
        except Exception as e:
            return {"uid": uid, "regime": regime, "accepted": 0, "failed": attempts,
                    "elapsed_s": 0.0, "error": "init_failed: %s" % str(e)[:80]}

        accepted = 0
        failed = 0
        errors = []
        t0 = time.time()
        for _ in range(attempts):
            if time.time() - t0 > timeout_s:
                errors.append("timeout")
                break
            try:
                generator.sample()
                accepted += 1
            except (FrequencyConstraintError, LexicalGapError):
                failed += 1
            except Exception as e:
                failed += 1
                err_str = str(e)[:60]
                if err_str not in errors:
                    errors.append(err_str)
        elapsed = round(time.time() - t0, 3)
        acceptance_rate = accepted / max(accepted + failed, 1)
        return {
            "uid": uid,
            "regime": regime,
            "accepted": accepted,
            "failed": failed,
            "attempts": accepted + failed,
            "elapsed_s": elapsed,
            "acceptance_rate": round(acceptance_rate, 3),
            "stall_risk": acceptance_rate < 0.1,
            "errors": "; ".join(errors[:3]),
        }
    finally:
        clear_sampling_policy()
        clear_query_caches()


def main():
    args = _parse_args()
    _apply_overlay_env(args)

    uids = args.uids or _available_uids()
    regimes = args.regimes

    fieldnames = ["uid", "regime", "accepted", "failed", "attempts",
                  "elapsed_s", "acceptance_rate", "stall_risk", "errors"]

    if args.output:
        outfile = open(args.output, "w", newline="")
    else:
        outfile = sys.stdout

    writer = csv.DictWriter(outfile, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    total = len(uids) * len(regimes)
    done = 0
    stall_count = 0
    for uid in uids:
        for regime in regimes:
            done += 1
            sys.stderr.write("[%d/%d] %s x %s ... " % (done, total, uid, regime))
            sys.stderr.flush()
            row = _probe_uid_regime(uid, regime, args.attempts, args.uid_timeout)
            writer.writerow(row)
            if hasattr(outfile, "flush"):
                outfile.flush()
            stall = row.get("stall_risk", False)
            if stall:
                stall_count += 1
            sys.stderr.write("acc=%.2f %s\n" % (row.get("acceptance_rate", 0), "STALL" if stall else "ok"))

    if args.output:
        outfile.close()

    sys.stderr.write("\nDone. %d/%d UID x regime pairs flagged as stall_risk.\n" % (stall_count, total))


if __name__ == "__main__":
    main()
