import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


FRAME_RE = re.compile(
    r"(ditrans_particle|trans_particle|intr_particle|ditrans_pp|intr_pp|trans|intr)(?:[:_(]([a-z]+)\)?)?"
)


def parse_flagged_report(path):
    removals = defaultdict(set)
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                index = int(parts[0])
            except ValueError:
                continue
            frame_field = parts[2]
            matches = list(FRAME_RE.finditer(frame_field))
            if not matches:
                continue
            for match in matches:
                frame_type = match.group(1)
                frame_value = match.group(2)
                removals[index].add((frame_type, frame_value))
    return removals


def frame_key(frame):
    frame_type = frame.get("type")
    if frame_type in {"intr_pp", "ditrans_pp"}:
        return frame_type, str(frame.get("prep", "")).strip().lower() or None
    if frame_type in {"intr_particle", "trans_particle", "ditrans_particle"}:
        return frame_type, str(frame.get("particle", "")).strip().lower() or None
    return frame_type, None


def prune_inventory(inventory, removals):
    pruned_entries = []
    summary = {
        "input_entries": len(inventory),
        "flagged_entry_indices": len(removals),
        "frames_removed": 0,
        "entries_dropped": 0,
        "entries_retained": 0,
    }
    for index, entry in enumerate(inventory):
        flagged = removals.get(index, set())
        new_frames = []
        for frame in entry.get("frames", []):
            if frame_key(frame) in flagged:
                summary["frames_removed"] += 1
                continue
            new_frames.append(frame)
        if new_frames:
            new_entry = dict(entry)
            new_entry["frames"] = new_frames
            pruned_entries.append(new_entry)
            summary["entries_retained"] += 1
        else:
            summary["entries_dropped"] += 1
    return pruned_entries, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--flags", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output")
    args = parser.parse_args()

    inventory_path = Path(args.inventory)
    flags_path = Path(args.flags)
    output_path = Path(args.output)
    summary_output_path = Path(args.summary_output) if args.summary_output else None

    inventory = json.loads(inventory_path.read_text())
    removals = parse_flagged_report(flags_path)
    pruned_entries, summary = prune_inventory(inventory, removals)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(pruned_entries, indent=2, sort_keys=True) + "\n")

    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
