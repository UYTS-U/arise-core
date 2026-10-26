"""Source and command contracts. No datasets, Torch imports, or model execution."""

import ast
import dataclasses
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_distribution(self):
        verifier = load_file("release_verifier", ROOT / "scripts/verify_release.py")
        self.assertEqual(verifier.verify()["status"], "PASS")

    def test_configuration_matches_runtime(self):
        config = load_file("release_config", ROOT / "src/arise/config.py")
        stored = json.loads((ROOT / "configs/training.json").read_text())
        self.assertEqual(dataclasses.asdict(config.Config()), stored["config"])
        registry = json.loads((ROOT / "configs/seeds.json").read_text())
        self.assertEqual(config.SEEDS, registry["seeds"])
        self.assertEqual(config.METHODS, ["full"])

    def test_public_cli_without_model_import(self):
        run = load_file("release_command", ROOT / "scripts/run.py")
        self.assertEqual(run.METHODS["arise"], "full")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run.py"), "--list"],
            capture_output=True,
            text=True,
            check=True,
        )
        contract = json.loads(result.stdout)
        self.assertEqual(contract["removal_datasets"], ["ml1m"])
        self.assertEqual(len(contract["main_datasets"]), 5)

    def test_removed_study_and_unregistered_seed_rejected_before_execution(self):
        before = sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*"))
        for args in [
            ["--dataset", "bluesky_a", "--method", "without-tl"],
            ["--dataset", "ml1m", "--seed", "2040"],
            ["--dataset", "ml1m", "--method", "rsir"],
        ]:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run.py")] + args,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
        self.assertEqual(before, sorted(str(p.relative_to(ROOT)) for p in ROOT.rglob("*")))

    def test_no_removed_config_attribute_is_read(self):
        config = load_file("release_fields", ROOT / "src/arise/config.py")
        fields = set(dataclasses.asdict(config.Config())) | {"dump"}
        for path in (ROOT / "src/arise").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    if node.value.id == "cfg":
                        self.assertIn(node.attr, fields, str(path))


if __name__ == "__main__":
    unittest.main()
