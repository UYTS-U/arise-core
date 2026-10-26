# ARISE research code. See NOTICE and CITATION.cff.
"""Frozen-state caches keyed by full weights and immutable input hashes."""

import hashlib, os
from pathlib import Path
import numpy as np
import torch
from .config import ROOT
from .data import pack, subset
from .cwm import make_frozen
from .util import progress


def model_hash(model):
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class FrozenCache:
    def __init__(self, model, env, d, condition, window, no_cold, out):
        inputs = (ROOT / "configs/data_manifest.json").read_bytes()
        version = (ROOT / "configs/release.json").read_bytes()
        key = hashlib.sha256(
            (
                "%s|%s|%d|%s|%d|%s"
                % (model_hash(model), env.name, d["stage"], condition, window, no_cold)
            ).encode()
            + inputs
            + version
        ).hexdigest()
        self.path = Path(out) / "cache" / key
        self.path.mkdir(parents=True, exist_ok=True)
        self.fields = ["v", "hL", "hS", "tag", "static"]
        if not (self.path / "READY").exists() or (self.path / "READY").read_text() != key:
            ng = len(d["targets"])
            maps = {
                k: np.lib.format.open_memmap(
                    self.path / (k + ".npy"),
                    mode="w+",
                    dtype=np.float32,
                    shape=(ng, 8 if k == "static" else model.cfg.dim),
                )
                for k in self.fields
            }
            model.eval()
            with torch.no_grad():
                e = model.items().detach()
                for off in range(0, len(d["user_ids"]), model.cfg.user_batch):
                    users = np.arange(off, min(off + model.cfg.user_batch, len(d["user_ids"])))
                    p = pack(d, users)
                    for start in range(0, len(p["groups"]), model.cfg.group_chunk):
                        q = subset(
                            p,
                            np.arange(
                                start, min(start + model.cfg.group_chunk, len(p["groups"]))
                            ),
                        )
                        b = make_frozen(model, env, d, q, condition, window, e, no_cold)
                        for k in self.fields:
                            maps[k][q["groups"]] = b[k].cpu().numpy()
                    progress(
                        out,
                        "cache frozen backbone",
                        stage=d["stage"],
                        condition=condition,
                        window=window,
                        users=min(off + model.cfg.user_batch, len(d["user_ids"])),
                        total_users=len(d["user_ids"]),
                    )
            for a in maps.values():
                a.flush()
            tmp = self.path / "READY.tmp"
            tmp.write_text(key)
            os.replace(str(tmp), str(self.path / "READY"))
        self.arrays = {k: np.load(self.path / (k + ".npy"), mmap_mode="r") for k in self.fields}

    def get(self, p, device):
        return {
            k: torch.as_tensor(np.array(a[p["groups"]]), device=device)
            for (k, a) in self.arrays.items()
        }
