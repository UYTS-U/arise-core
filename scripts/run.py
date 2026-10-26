"""Launch a single registered core run; --list never imports the model."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "arise": "full",
    "without-tl": "no_cold",
    "without-cwm": "no_adaptive",
    "without-ct": "no_cooc",
    "without-cwm-ct": "no_adaptive_cooc",
}


def main():
    seeds = json.loads((ROOT / "configs/seeds.json").read_text())["seeds"]
    datasets = json.loads((ROOT / "configs/release.json").read_text())["main_datasets"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List supported runs without execution"
    )
    parser.add_argument("--dataset", choices=datasets)
    parser.add_argument("--method", choices=METHODS, default="arise")
    parser.add_argument("--seed", type=int, choices=seeds, default=seeds[0])
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                {
                    "main_datasets": datasets,
                    "methods": list(METHODS),
                    "removal_datasets": ["ml1m"],
                    "training_seeds": seeds,
                },
                indent=2,
            )
        )
        return
    if args.dataset is None:
        parser.error("--dataset is required unless --list is used")
    if args.method != "arise":
        if args.dataset != "ml1m":
            parser.error("component removals are restricted to ml1m")
        reference = ROOT / "results/ml1m" / str(args.seed) / "full/selection.json"
        if not reference.is_file() or not (reference.parent / "done.json").is_file():
            parser.error("run the same-seed ARISE reference before a component removal")
    if not (ROOT / "data" / args.dataset / "READY.json").is_file():
        parser.error("prepared inputs are required; see docs/INPUTS.md")
    code = (
        "from arise.runner import Worker; import sys; "
        "Worker(sys.argv[1], sys.argv[2], int(sys.argv[3])).run()"
    )
    command = [sys.executable, "-c", code, args.dataset, METHODS[args.method], str(args.seed)]
    subprocess.run(
        command, cwd=str(ROOT), env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), check=True
    )


if __name__ == "__main__":
    main()
