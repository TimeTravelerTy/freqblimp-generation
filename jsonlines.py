import builtins
import json


class Writer:
    def __init__(self, fp, flush=False):
        self._fp = fp
        self._flush = flush

    def write(self, obj):
        self._fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
        if self._flush:
            self._fp.flush()

    def write_all(self, iterable):
        for obj in iterable:
            self.write(obj)

    def close(self):
        if hasattr(self._fp, "close"):
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class Reader:
    def __init__(self, fp):
        self._fp = fp

    def __iter__(self):
        for line in self._fp:
            line = line.strip()
            if line:
                yield json.loads(line)

    def close(self):
        if hasattr(self._fp, "close"):
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def open(path, mode="r", encoding="utf-8"):
    fp = builtins.open(path, mode, encoding=encoding)
    if "r" in mode:
        return Reader(fp)
    return Writer(fp)
