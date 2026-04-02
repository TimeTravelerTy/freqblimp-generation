import argparse
import importlib
import inspect
import os
from pathlib import Path

from utils import data_generator
from utils.randomize import SamplingPolicy, configure_sampling_policy


def _available_uids():
    root = Path(__file__).resolve().parent
    ignored = {"__init__", "run", "build_overlay", "sbatch_generator"}
    return sorted(
        path.stem
        for path in root.glob("*.py")
        if path.stem not in ignored
    )


def _apply_runtime_env(args):
    os.environ["FREQBLIMP_USE_OVERLAY"] = "1" if args.use_overlay else "0"
    if args.overlay_path:
        os.environ["FREQBLIMP_VOCAB_OVERLAY"] = args.overlay_path
    if args.overlay_manifest_path:
        os.environ["FREQBLIMP_OVERLAY_MANIFEST"] = args.overlay_manifest_path
    if args.frequency_cache_path:
        os.environ["FREQBLIMP_FREQUENCY_CACHE"] = args.frequency_cache_path


def _policy_from_args(args):
    controlled_pos = args.controlled_pos or ("noun", "verb", "adjective")
    zipf_min = {
        "noun": args.noun_zipf_min if args.noun_zipf_min is not None else args.zipf_min_all,
        "verb": args.verb_zipf_min if args.verb_zipf_min is not None else args.zipf_min_all,
        "adjective": args.adj_zipf_min if args.adj_zipf_min is not None else args.zipf_min_all,
    }
    zipf_max = {
        "noun": args.noun_zipf_max if args.noun_zipf_max is not None else args.zipf_max_all,
        "verb": args.verb_zipf_max if args.verb_zipf_max is not None else args.zipf_max_all,
        "adjective": args.adj_zipf_max if args.adj_zipf_max is not None else args.zipf_max_all,
    }
    return SamplingPolicy(
        seed=args.seed,
        controlled_pos=controlled_pos,
        zipf_min=zipf_min,
        zipf_max=zipf_max,
        zipf_weighted_sampling=args.zipf_weighted_sampling,
        zipf_temp=args.zipf_temp,
        overlay_enabled=args.use_overlay,
        overlay_path=args.overlay_path,
        overlay_manifest_path=args.overlay_manifest_path,
        frequency_cache_path=args.frequency_cache_path,
    )


def _build_generator(module):
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


def _run_uid(uid, args):
    module = importlib.import_module("generation_projects.blimp.%s" % uid)
    configure_sampling_policy(_policy_from_args(args))
    generator = _build_generator(module)
    if args.output_path:
        generator.generate_paradigm(number_to_generate=args.number_to_generate, absolute_path=args.output_path)
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.generate_paradigm(
        number_to_generate=args.number_to_generate,
        absolute_path=str(output_dir / ("%s.jsonl" % generator.uid)),
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Run BLiMP generators with frequency-aware sampling.")
    parser.add_argument("uid", help="Paradigm UID or 'all'.")
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    _apply_runtime_env(args)
    if args.uid == "all":
        if args.output_path:
            parser.error("--output-path only supports a single uid.")
        for uid in _available_uids():
            _run_uid(uid, args)
        return
    _run_uid(args.uid, args)


if __name__ == "__main__":
    main()
