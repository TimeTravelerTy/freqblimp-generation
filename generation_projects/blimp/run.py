import argparse
import concurrent.futures
import importlib
import inspect
import json
import os
import re
import time
import traceback
from pathlib import Path

from utils.randomize import SamplingPolicy, configure_sampling_policy, set_trace_recording_enabled


_DECLARED_UID_CACHE = {}
_DECLARED_UID_PATTERN = re.compile(r"""uid\s*=\s*['"]([^'"]+)['"]""")


def _data_generator_module():
    from utils import data_generator
    return data_generator


def _clear_query_caches():
    from utils.vocab_table import clear_query_caches
    clear_query_caches()


def _rss_mb():
    """Return current RSS in MiB. Reads /proc/self/status on Linux; falls back to ru_maxrss."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MiB
    except Exception:
        pass
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kB on Linux
    import sys
    return ru / 1024 if sys.platform != "darwin" else ru / 1024 / 1024


def _available_uids():
    root = Path(__file__).resolve().parent
    ignored = {"__init__", "run", "build_overlay", "sbatch_generator"}
    return sorted(
        path.stem
        for path in root.glob("*.py")
        if path.stem not in ignored
    )


def _declared_uid_for_module(uid):
    if uid not in _DECLARED_UID_CACHE:
        declared_uid = uid
        module_path = Path(__file__).resolve().parent / ("%s.py" % uid)
        try:
            text = module_path.read_text()
        except Exception:
            text = ""
        match = _DECLARED_UID_PATTERN.search(text)
        if match:
            declared_uid = match.group(1)
        _DECLARED_UID_CACHE[uid] = declared_uid
    return _DECLARED_UID_CACHE[uid]


def _predicted_output_path(uid, config):
    if config["output_path"]:
        return config["output_path"]
    output_dir = Path(config["output_dir"])
    return str(output_dir / ("%s.jsonl" % _declared_uid_for_module(uid)))


def _is_nonempty_output(path):
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


def _runtime_config_from_args(args):
    return {
        "number_to_generate": args.number_to_generate,
        "output_dir": args.output_dir,
        "output_path": args.output_path,
        "seed": args.seed,
        "controlled_pos": tuple(args.controlled_pos or ("noun", "verb", "adjective")),
        "zipf_min_all": args.zipf_min_all,
        "zipf_max_all": args.zipf_max_all,
        "noun_zipf_min": args.noun_zipf_min,
        "noun_zipf_max": args.noun_zipf_max,
        "verb_zipf_min": args.verb_zipf_min,
        "verb_zipf_max": args.verb_zipf_max,
        "adj_zipf_min": args.adj_zipf_min,
        "adj_zipf_max": args.adj_zipf_max,
        "zipf_weighted_sampling": args.zipf_weighted_sampling,
        "zipf_temp": args.zipf_temp,
        "use_overlay": args.use_overlay,
        "overlay_path": args.overlay_path,
        "overlay_manifest_path": args.overlay_manifest_path,
        "frequency_cache_path": args.frequency_cache_path,
        "record_trace": not args.no_trace,
        "show_progress": not args.no_progress and args.jobs == 1,
        "continue_on_error": args.continue_on_error,
        "skip_existing": args.skip_existing,
        "jobs": args.jobs,
    }


def _apply_runtime_env(config):
    os.environ["FREQBLIMP_USE_OVERLAY"] = "1" if config["use_overlay"] else "0"
    os.environ["FREQBLIMP_SHOW_PROGRESS"] = "1" if config["show_progress"] else "0"
    if config["overlay_path"]:
        os.environ["FREQBLIMP_VOCAB_OVERLAY"] = config["overlay_path"]
    if config["overlay_manifest_path"]:
        os.environ["FREQBLIMP_OVERLAY_MANIFEST"] = config["overlay_manifest_path"]
    if config["frequency_cache_path"]:
        os.environ["FREQBLIMP_FREQUENCY_CACHE"] = config["frequency_cache_path"]
    set_trace_recording_enabled(config["record_trace"])


def _policy_from_config(config):
    controlled_pos = config["controlled_pos"] or ("noun", "verb", "adjective")
    zipf_min = {
        "noun": config["noun_zipf_min"] if config["noun_zipf_min"] is not None else config["zipf_min_all"],
        "verb": config["verb_zipf_min"] if config["verb_zipf_min"] is not None else config["zipf_min_all"],
        "adjective": config["adj_zipf_min"] if config["adj_zipf_min"] is not None else config["zipf_min_all"],
    }
    zipf_max = {
        "noun": config["noun_zipf_max"] if config["noun_zipf_max"] is not None else config["zipf_max_all"],
        "verb": config["verb_zipf_max"] if config["verb_zipf_max"] is not None else config["zipf_max_all"],
        "adjective": config["adj_zipf_max"] if config["adj_zipf_max"] is not None else config["zipf_max_all"],
    }
    return SamplingPolicy(
        seed=config["seed"],
        controlled_pos=controlled_pos,
        zipf_min=zipf_min,
        zipf_max=zipf_max,
        zipf_weighted_sampling=config["zipf_weighted_sampling"],
        zipf_temp=config["zipf_temp"],
        overlay_enabled=config["use_overlay"],
        overlay_path=config["overlay_path"],
        overlay_manifest_path=config["overlay_manifest_path"],
        frequency_cache_path=config["frequency_cache_path"],
    )


def _build_generator(module):
    data_generator = _data_generator_module()
    if hasattr(module, "build_generator"):
        return module.build_generator()
    generator_classes = []
    for _name, value in inspect.getmembers(module, inspect.isclass):
        if value in {
            data_generator.Generator,
            data_generator.BenchmarkGenerator,
            data_generator.ScalarImplicatureGenerator,
            data_generator.PresuppositionGenerator,
        }:
            continue
        if issubclass(value, data_generator.Generator):
            generator_classes.append(value)
    if not generator_classes:
        raise RuntimeError("No generator class found in module %s" % module.__name__)
    if len(generator_classes) > 1:
        generator_classes.sort(key=lambda cls: cls.__name__)
    return generator_classes[0]()


def _run_uid(uid, config):
    started = time.time()
    predicted_output_path = _predicted_output_path(uid, config)
    if config.get("skip_existing") and _is_nonempty_output(predicted_output_path):
        print("[skip-existing] %s: %s (%d bytes)" % (uid, predicted_output_path, os.path.getsize(predicted_output_path)))
        return {
            "uid": uid,
            "status": "skipped",
            "duration_seconds": round(time.time() - started, 3),
            "output_path": predicted_output_path,
        }
    module = importlib.import_module("generation_projects.blimp.%s" % uid)
    configure_sampling_policy(_policy_from_config(config))
    generator = _build_generator(module)
    if config["output_path"]:
        if config.get("skip_existing") and _is_nonempty_output(config["output_path"]):
            print("[skip-existing] %s: %s (%d bytes)" % (uid, config["output_path"], os.path.getsize(config["output_path"])))
            return {
                "uid": uid,
                "status": "skipped",
                "duration_seconds": round(time.time() - started, 3),
                "output_path": config["output_path"],
            }
        generator.generate_paradigm(number_to_generate=config["number_to_generate"], absolute_path=config["output_path"])
        return {
            "uid": uid,
            "status": "ok",
            "duration_seconds": round(time.time() - started, 3),
            "output_path": config["output_path"],
        }
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / ("%s.jsonl" % generator.uid))
    if config.get("skip_existing") and _is_nonempty_output(output_path):
        print("[skip-existing] %s: %s (%d bytes)" % (uid, output_path, os.path.getsize(output_path)))
        return {
            "uid": uid,
            "status": "skipped",
            "duration_seconds": round(time.time() - started, 3),
            "output_path": output_path,
        }
    generator.generate_paradigm(
        number_to_generate=config["number_to_generate"],
        absolute_path=output_path,
    )
    return {
        "uid": uid,
        "status": "ok",
        "duration_seconds": round(time.time() - started, 3),
        "output_path": output_path,
    }


def _batched_uids(requested_uids, jobs):
    parallelism = max(1, min(jobs, len(requested_uids)))
    batches = [[] for _ in range(parallelism)]
    for idx, uid in enumerate(requested_uids):
        batches[idx % parallelism].append(uid)
    return [batch for batch in batches if batch]


def _run_uid_batch(uids, config):
    _apply_runtime_env(config)
    results = []
    pid = os.getpid()
    for uid in uids:
        try:
            result = _run_uid(uid, config)
            results.append(result)
        except Exception as exc:
            if not config["continue_on_error"]:
                raise
            print("[continue-on-error] %s failed: %s" % (uid, exc))
            results.append(
                {
                    "uid": uid,
                    "status": "error",
                    "duration_seconds": None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        finally:
            _clear_query_caches()
            print("[mem pid=%d] after %s: %.0f MiB RSS" % (pid, uid, _rss_mb()), flush=True)
    return results


def _write_summary(config, requested_uids, results):
    if config["output_path"]:
        return
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "requested_uids": list(requested_uids),
        "continue_on_error": config["continue_on_error"],
        "jobs": config["jobs"],
        "ok_count": sum(1 for item in results if item.get("status") == "ok"),
        "error_count": sum(1 for item in results if item.get("status") == "error"),
        "results": results,
    }
    with open(output_dir / "run_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _requested_uids(args):
    if "all" in args.uids:
        return _available_uids()
    return args.uids


def build_parser():
    parser = argparse.ArgumentParser(description="Run BLiMP generators with frequency-aware sampling.")
    parser.add_argument("uids", nargs="+", help="One or more paradigm UIDs, or 'all'.")
    parser.add_argument("--number-to-generate", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/blimp")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controlled-pos", nargs="+", default=("noun", "verb", "adjective"))
    parser.add_argument("--zipf-min-all", type=float, default=None)
    parser.add_argument("--zipf-max-all", type=float, default=None)
    parser.add_argument("--noun-zipf-min", type=float, default=None)
    parser.add_argument("--noun-zipf-max", type=float, default=None)
    parser.add_argument("--verb-zipf-min", type=float, default=None)
    parser.add_argument("--verb-zipf-max", type=float, default=None)
    parser.add_argument("--adj-zipf-min", type=float, default=None)
    parser.add_argument("--adj-zipf-max", type=float, default=None)
    parser.add_argument("--zipf-weighted-sampling", action="store_true")
    parser.add_argument("--zipf-temp", type=float, default=1.0)
    parser.add_argument("--use-overlay", action="store_true")
    parser.add_argument("--overlay-path", default="vocabulary_overlay.csv")
    parser.add_argument("--overlay-manifest-path", default="vocabulary_overlay_manifest.json")
    parser.add_argument("--frequency-cache-path", default="outputs/cache/vocabulary_frequency_cache.json")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-trace", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip paradigms whose output file already exists and is non-empty.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    requested_uids = _requested_uids(args)
    config = _runtime_config_from_args(args)
    if args.output_path and len(requested_uids) != 1:
        parser.error("--output-path only supports a single uid.")
    if args.output_path and args.jobs != 1:
        parser.error("--output-path does not support --jobs > 1.")
    if args.jobs < 1:
        parser.error("--jobs must be at least 1.")
    if args.jobs == 1 or len(requested_uids) == 1:
        _apply_runtime_env(config)
        results = []
        pid = os.getpid()
        for uid in requested_uids:
            try:
                results.append(_run_uid(uid, config))
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print("[continue-on-error] %s failed: %s" % (uid, exc))
                results.append(
                    {
                        "uid": uid,
                        "status": "error",
                        "duration_seconds": None,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                _clear_query_caches()
                print("[mem pid=%d] after %s: %.0f MiB RSS" % (pid, uid, _rss_mb()), flush=True)
        _write_summary(config, requested_uids, results)
        return
    batches = _batched_uids(requested_uids, args.jobs)
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(batches)) as executor:
        futures = [executor.submit(_run_uid_batch, batch, config) for batch in batches]
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda item: requested_uids.index(item["uid"]))
    _write_summary(config, requested_uids, results)


if __name__ == "__main__":
    main()
