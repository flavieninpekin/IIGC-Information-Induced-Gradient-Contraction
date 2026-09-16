"""One-off migration: add canonical kappa_mix to legacy component JSONs.

Legacy files store only the aggregate components (E_shared, E_contrast,
sigma2). Under the canonical schema

    kappa_mix = E_shared / (E_shared + E_contrast),

so the structural metric can be recovered without rerunning any rollout. This
script only adds fields; it never deletes or rewrites existing numbers.

Usage:
  python backfill_kappa_mix.py [--check]
"""

import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", ".."))
TARGETS = [
    os.path.join(ROOT, "data", "kappa", "server_tasks", "results",
                 "oc_field_axis.json"),
    os.path.join(ROOT, "data", "kappa", "server_tasks", "results",
                 "oc_switch_kappa.json"),
]
SCHEMA = "mixture-v1"


def backfill_entry(entry):
    """Return True when the entry gained a kappa_mix field."""
    if not isinstance(entry, dict):
        return False
    if "E_shared" not in entry or "E_contrast" not in entry:
        return False
    if "kappa_mix" in entry:
        return False
    denom = entry["E_shared"] + entry["E_contrast"]
    entry["kappa_mix"] = (entry["E_shared"] / denom) if denom > 0 else 0.0
    entry["kappa_schema"] = SCHEMA
    return True


def process(path, check=False):
    with open(path) as f:
        data = json.load(f)
    changed = 0
    for key, value in data.items():
        if key == "_metadata":
            continue
        if isinstance(value, dict) and "E_shared" in value:
            if backfill_entry(value):
                changed += 1
        else:
            for sub in value.values() if isinstance(value, dict) else []:
                if backfill_entry(sub):
                    changed += 1
    if not check and changed:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=float)
    print(f"{os.path.basename(path)}: {changed} entries updated "
          f"({'check only' if check else 'saved'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    for path in TARGETS:
        if os.path.exists(path):
            process(path, check=args.check)
        else:
            print(f"skip missing {path}")


if __name__ == "__main__":
    main()
