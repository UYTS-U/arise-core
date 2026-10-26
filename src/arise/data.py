# ARISE research code. See NOTICE and CITATION.cff.
"""Read exported B and public inputs; never open hidden A or ratings."""

import numpy as np
from scipy import sparse
import torch
from .config import ROOT, load_spec

CONDITIONS = ["clean", "deleted60", "replaced60"]


def public_view(d):
    """No target/negative packets, hidden profile, ratings or future sequence."""
    return {
        k: v
        for (k, v) in d.items()
        if k == "round_index" or k == "address" or k.startswith("tags_")
    }


def split_masks(timestamp):
    ts = np.asarray(timestamp).reshape(-1, 5)
    cut = int(np.sort(ts.ravel())[int(0.8 * ts.size)])
    return (ts[:, -1] < cut, ts[:, 0] >= cut, cut)


class Environment:
    def __init__(self, name, device="cpu"):
        self.name = name
        self.spec = load_spec(name)
        self.path = ROOT / "data" / name
        self.device = torch.device(device)
        self.x = sparse.load_npz(self.path / "features.npz").astype(np.float32)
        self.text = torch.as_tensor(
            np.array(np.load(self.path / "text.npy", mmap_mode="r")), device=self.device
        )
        coo = self.x.tocoo()
        self.tx = torch.sparse_coo_tensor(
            np.stack([coo.row, coo.col]), coo.data, coo.shape, device=self.device
        ).coalesce()
        self.created = np.load(self.path / "created.npy", mmap_mode="r")
        self.nitems = self.x.shape[0]
        self.address_features = None
        if (self.path / "address_features.npy").exists():
            self.address_features = torch.as_tensor(
                np.load(self.path / "address_features.npy"), device=self.device
            )
        self.tie = np.random.default_rng(2040).permutation(self.nitems)
        self.priority = np.empty(self.nitems, np.int64)
        self.priority[self.tie] = np.arange(self.nitems)

    def stage(self, k):
        p = self.path / ("D%d" % k)
        d = {f.stem: np.load(f, mmap_mode="r") for f in p.glob("*.npy")}
        (d["training"], d["validation"], d["cut"]) = split_masks(d["timestamp"])
        d["stage"] = k
        d["_visibility"] = self.spec["visibility"]
        return d

    def project(self, weight):
        return torch.sparse.mm(self.tx, weight)


def selected_users(d, mask):
    return np.unique(d["round_user"][mask])


def pack(d, users, mask=None, width=100):
    groups = []
    owners = []
    histories = []
    lengths = []
    timestamps = []
    seen = []
    starts = []
    for u in users:
        allg = np.arange(int(d["offsets"][u]) // 5, int(d["offsets"][u + 1]) // 5)
        gs = allg if mask is None else allg[mask[allg]]
        if not len(gs):
            continue
        owner = len(starts)
        starts.append(len(groups))
        offset = int(d["offsets"][u])
        for g in gs:
            n = 5 * int(d["round_index"][g])
            context = d["item"][offset : offset + n]
            recent = context[-width:]
            h = np.zeros(width, np.int64)
            h[: len(recent)] = recent
            t = np.zeros(width, np.int64)
            t[: len(recent)] = d["timestamp"][offset : offset + n][-width:]
            groups.append(g)
            owners.append(owner)
            histories.append(h)
            lengths.append(len(recent))
            timestamps.append(t)
            seen.append(d["item"][offset : offset + n])
    return dict(
        groups=np.asarray(groups, np.int64),
        owner=np.asarray(owners, np.int64),
        users=np.asarray(users, np.int64),
        starts=np.asarray(starts, np.int64),
        history=np.asarray(histories, np.int64),
        lengths=np.asarray(lengths, np.int64),
        timestamps=np.asarray(timestamps, np.int64),
        seen=seen,
    )


def subset(p, ix):
    q = dict(p)
    for k in ["groups", "owner", "history", "lengths", "timestamps"]:
        q[k] = p[k][ix]
    q["seen"] = [p["seen"][int(i)] for i in ix]
    return q


def allowed(env, d, p):
    cat = d["catalog"]
    out = np.ones((len(p["groups"]), len(cat)), bool)
    if env.spec["visibility"] != "stage":
        t = d["round_end"] if env.spec["visibility"] == "end" else d["round_start"]
        out &= env.created[cat][None, :] <= t[p["groups"], None]
    if not env.spec["revisit"]:
        lookup = np.full(env.nitems, -1, np.int64)
        lookup[cat] = np.arange(len(cat))
        for r, items in enumerate(p["seen"]):
            x = lookup[items]
            out[r, x[x >= 0]] = False
    return torch.as_tensor(out, device=env.device)


def tag_arrays(d, condition):
    return [d["tags_" + condition]] + ([d["address"]] if "address" in d else [])
