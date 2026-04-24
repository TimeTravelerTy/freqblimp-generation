import importlib
import inspect
import re
from pathlib import Path


_DECLARED_UID_PATTERN = re.compile(r"""uid\s*=\s*['"]([^'"]+)['"]""")
_HELPER_STEMS = frozenset(
    {
        "__init__",
        "build_overlay",
        "overlay_coverage",
        "overlay_guards",
        "registry",
        "run",
    }
)


def _module_root():
    return Path(__file__).resolve().parent


def helper_stems():
    return set(_HELPER_STEMS)


def generator_stems():
    return sorted(
        path.stem
        for path in _module_root().glob("*.py")
        if path.stem not in _HELPER_STEMS
    )


def declared_uid_for_stem(stem):
    module_path = _module_root() / ("%s.py" % stem)
    try:
        text = module_path.read_text()
    except Exception:
        return stem
    match = _DECLARED_UID_PATTERN.search(text)
    return match.group(1) if match else stem


def stem_alias_map():
    aliases = {}
    for stem in generator_stems():
        aliases[stem] = stem
        aliases[declared_uid_for_stem(stem)] = stem
    return aliases


def resolve_requested_stems(requested):
    aliases = stem_alias_map()
    resolved = []
    for item in requested:
        stem = aliases.get(item)
        if stem is None:
            raise KeyError(item)
        resolved.append(stem)
    return resolved


def build_generator_from_stem(stem):
    from utils import data_generator

    module = importlib.import_module("generation_projects.blimp.%s" % stem)
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
    generator_classes.sort(key=lambda cls: cls.__name__)
    return generator_classes[0]()
