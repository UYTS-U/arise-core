"""Validate external prepared inputs without importing Torch or starting training."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check(dataset, verify_hashes=False):
    import numpy as np
    from scipy import sparse

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    root = ROOT / "data" / dataset
    spec = json.loads((ROOT / "configs/datasets.json").read_text())[dataset]
    require(root.is_dir(), "Prepare external inputs first; see docs/INPUTS.md")
    for name in ["features.npz", "text.npy", "created.npy", "READY.json"]:
        require((root / name).is_file(), "Missing " + name)
    n = sparse.load_npz(root / "features.npz").shape[0]
    text = np.load(root / "text.npy", mmap_mode="r", allow_pickle=False)
    created = np.load(root / "created.npy", mmap_mode="r", allow_pickle=False)
    require(text.ndim == 2 and len(text) == n, "text.npy must have one row per item")
    require(created.shape == (n,), "created.npy must have one time per item")
    if dataset == "yelp":
        city = np.load(root / "address_features.npy", mmap_mode="r", allow_pickle=False)
        require(city.shape == (spec["tag_sizes"][1], 6), "Invalid public city features")
    names = [
        "user_ids",
        "offsets",
        "item",
        "timestamp",
        "round_user",
        "round_index",
        "targets",
        "negatives",
        "catalog",
        "round_start",
        "round_end",
        "tags_clean",
        "tags_deleted60",
        "tags_replaced60",
    ]
    if dataset == "yelp":
        names.append("address")
    cohorts = set()
    for stage in range(4):
        folder = root / ("D%d" % stage)
        arrays = {
            name: np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            for name in names
        }
        users = arrays["user_ids"]
        u, g = len(users), len(arrays["targets"])
        require(u > 0 and g > 0, "Empty cohort or groups")
        require(len(np.unique(users)) == u, "Duplicate user within a stage")
        ids = set(users.tolist())
        require(not cohorts.intersection(ids), "Users must belong to disjoint stages")
        cohorts.update(ids)
        offsets = arrays["offsets"]
        require(
            offsets.shape == (u + 1,) and offsets[0] == 0 and offsets[-1] == 5 * g,
            "Offsets must span complete five-event groups",
        )
        counts = np.diff(offsets)
        require(np.all(counts >= 10) and np.all(counts % 5 == 0), "Invalid B lengths")
        require(
            arrays["item"].shape == (5 * g,) and arrays["timestamp"].size == 5 * g,
            "B item/timestamp lengths disagree",
        )
        require(
            arrays["targets"].shape == (g, 5) and arrays["negatives"].shape == (g, 15),
            "Expected five targets and fifteen sampled controls per group",
        )
        owner = np.repeat(np.arange(u), counts // 5)
        index = np.concatenate([np.arange(length // 5) for length in counts])
        require(np.array_equal(arrays["round_user"], owner), "Group ownership mismatch")
        require(np.array_equal(arrays["round_index"], index), "Group index mismatch")
        require(np.array_equal(arrays["targets"].ravel(), arrays["item"]), "Targets/B mismatch")
        for name in ["item", "targets", "negatives", "catalog"]:
            value = arrays[name]
            require(np.issubdtype(value.dtype, np.integer), name + " must contain integer IDs")
            require(
                value.size > 0 and value.min() >= 0 and value.max() < n,
                name + " contains out-of-range items",
            )
        catalog = arrays["catalog"]
        require(
            catalog.ndim == 1 and len(np.unique(catalog)) == len(catalog),
            "Candidate catalog must be a unique one-dimensional list",
        )
        for name in ["round_start", "round_end"]:
            require(arrays[name].shape == (g,), name + " must align with groups")
        for name in ["tags_clean", "tags_deleted60", "tags_replaced60"] + (
            ["address"] if dataset == "yelp" else []
        ):
            tags = arrays[name]
            size = spec["tag_sizes"][1 if name == "address" else 0]
            require(
                tags.ndim == 2 and tags.shape[0] == u and tags.shape[1] > 0,
                name + " must have one padded declaration row per user",
            )
            require(
                np.issubdtype(tags.dtype, np.integer) and np.all((tags >= -1) & (tags < size)),
                name + " contains invalid declaration IDs",
            )
    checked = 0
    if verify_hashes:
        manifest = json.loads((ROOT / "configs/data_manifest.json").read_text())[dataset]
        for relative, expected in manifest.items():
            path = root / relative
            require(
                path.is_file() and digest(path) == expected, "Input hash mismatch: " + relative
            )
            checked += 1
    return {"status": "PASS", "dataset": dataset, "items": n, "checked_hashes": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=["ml1m", "ml10m", "zhihu", "bluesky_a", "yelp"]
    )
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.dataset, args.verify_hashes)))


if __name__ == "__main__":
    main()
