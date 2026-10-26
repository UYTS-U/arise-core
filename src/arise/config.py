# ARISE research code. See NOTICE and CITATION.cff.
from dataclasses import dataclass, asdict
from pathlib import Path
import json

SEEDS = [48509147, 868610493, 1514168990]
DATASETS = ["zhihu", "bluesky_a", "ml1m", "ml10m", "yelp"]
METHODS = ["full"]
ABLATIONS = ["no_cold", "no_adaptive", "no_cooc", "no_adaptive_cooc"]
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    seed: int = 2040
    dim: int = 64
    layers: int = 2
    heads: int = 4
    max_history: int = 100
    dropout: float = 0.1
    pi0: float = 0.3
    gamma: float = 0.35
    feedback_calibration: bool = False
    tau: float = 0.5
    lr: float = 0.001
    weight_decay: float = 0.0001
    user_batch: int = 64
    group_chunk: int = 128
    d0_epochs: int = 100
    incremental_epochs: int = 20
    minimum_epochs: int = 3
    patience: int = 2
    stability: float = 10.0
    gate_stability: float = 10.0
    rank: int = 8

    def dump(self):
        return asdict(self)


def load_spec(name):
    return json.loads((ROOT / "configs/datasets.json").read_text())[name]
