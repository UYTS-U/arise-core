"""Check the distributed files and local import closure using only the standard library."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRS = {
    "data",
    "results",
    "cache",
    "logs",
    ".venv",
    "build",
    "dist",
    ".git",
    ".ruff_cache",
}


def distributed_files(root):
    result = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (
            relative.parts[0] in RUNTIME_DIRS
            or "__pycache__" in relative.parts
            or any(part.endswith(".egg-info") for part in relative.parts)
        ):
            continue
        if path.is_symlink():
            raise ValueError(
                "Symlinks are not permitted in distributed files: " + str(relative)
            )
        if path.is_file():
            result.add(relative.as_posix())
    return result


def check_imports(root, relative, tree):
    if not relative.startswith("src/arise/"):
        return
    module = relative[4:-3].replace("/", ".")
    package = module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            target = importlib.util.resolve_name(
                "." * node.level + (node.module or ""), package
            )
            stem = root / "src" / target.replace(".", "/")
            if not stem.with_suffix(".py").is_file() and not (stem / "__init__.py").is_file():
                raise ValueError("Unresolved local import in " + relative + ": " + target)


def verify(root=ROOT):
    manifest = {}
    for line in (root / "MANIFEST.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        parts = PurePosixPath(relative)
        if (
            not re.fullmatch("[0-9a-f]{64}", digest)
            or parts.is_absolute()
            or ".." in parts.parts
            or relative in manifest
        ):
            raise ValueError("Invalid manifest entry: " + relative)
        manifest[relative] = digest
    actual = distributed_files(root) - {"MANIFEST.sha256"}
    if actual != set(manifest):
        raise ValueError(
            "File set mismatch: " + repr(sorted(actual.symmetric_difference(manifest)))
        )
    for relative, expected in manifest.items():
        data = (root / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Checksum mismatch: " + relative)
        if relative.endswith(".py"):
            tree = ast.parse(data.decode("utf-8"), filename=relative)
            check_imports(root, relative, tree)
        # Input fingerprints are text; actual data, weights and archives are excluded.
        if Path(relative).suffix.lower() in {
            ".npy",
            ".npz",
            ".pt",
            ".pth",
            ".zip",
            ".csv",
            ".gz",
        }:
            raise ValueError("Non-source artifact included: " + relative)
        if re.search(rb"/(?:Users|home|root)/|127\.0\.0\.1", data):
            raise ValueError("Private environment trace in " + relative)
    seeds = json.loads((root / "configs/seeds.json").read_text())
    if seeds["seeds"] != [48509147, 868610493, 1514168990] or seeds["data_seed"] != 2040:
        raise ValueError("Unexpected seed registry")
    release = json.loads((root / "configs/release.json").read_text())
    if release["component_removal_datasets"] != ["ml1m"]:
        raise ValueError("Unexpected removal scope")
    return {
        "status": "PASS",
        "distributed_files": len(manifest) + 1,
        "local_imports": "resolved",
        "data_included": False,
        "training_run": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
