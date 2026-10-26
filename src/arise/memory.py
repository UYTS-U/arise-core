# ARISE research code. See NOTICE and CITATION.cff.
"""Cumulative sufficient statistics. Past user-event matrices are discarded."""

from pathlib import Path
import numpy as np
from scipy import sparse


class Memory:
    def __init__(self, n, with_ct=True):
        self.C = sparse.csr_matrix((n, n), dtype=np.float64)
        self.nu = np.zeros(n, np.float64)
        self.written = []
        self.with_ct = with_ct

    def update(self, d, mask):
        assert d["stage"] not in self.written, "Each stage may write only once"
        n = len(self.nu)
        ru = np.bincount(d["round_user"][mask], minlength=len(d["user_ids"]))
        ids = np.flatnonzero(mask)
        self.nu += np.bincount(d["targets"][mask].ravel(), minlength=n)
        for off in range(0, len(ids) if self.with_ct else 0, 8192):
            rr = []
            cc = []
            vv = []
            for g in ids[off : off + 8192]:
                u = int(d["round_user"][g])
                r = int(d["round_index"][g])
                o = int(d["offsets"][u])
                hist = d["item"][o + max(0, 5 * r - 50) : o + 5 * r]
                if not len(hist):
                    continue
                rr.extend(np.repeat(hist, 5))
                cc.extend(np.tile(d["targets"][g], len(hist)))
                vv.extend([1 / (ru[u] * len(hist) * 5)] * (len(hist) * 5))
            self.C += sparse.coo_matrix((vv, (rr, cc)), shape=(n, n)).tocsr()
        self.written.append(d["stage"])

    def kernel(self, catalog):
        known = np.unique(np.r_[np.flatnonzero(self.nu), catalog])
        p = (self.nu + 1) / (self.nu[known].sum() + len(known))
        c = self.C.tocoo()
        v = np.maximum(
            0, np.log1p(c.data / p[c.col]) - np.log1p(np.asarray(self.C.sum(1)).ravel()[c.row])
        )
        R = sparse.csr_matrix((v, (c.row, c.col)), shape=c.shape)
        R.eliminate_zeros()
        return R[:, catalog]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(path / "ct.npz", self.C)
        np.savez(path / "state.npz", nu=self.nu, written=self.written, with_ct=self.with_ct)

    @classmethod
    def load(cls, path):
        path = Path(path)
        z = np.load(path / "state.npz")
        m = cls(len(z["nu"]), bool(z["with_ct"]))
        m.nu = z["nu"]
        m.written = z["written"].tolist()
        m.C = sparse.load_npz(path / "ct.npz")
        return m


def read(p, R, window=50):
    rr = []
    cc = []
    vv = []
    for j, (h, n) in enumerate(zip(p["history"], p["lengths"])):
        ids = h[max(0, int(n) - window) : int(n)]
        if len(ids):
            rr.extend([j] * len(ids))
            cc.extend(ids)
            vv.extend([1 / len(ids)] * len(ids))
    return (
        (sparse.csr_matrix((vv, (rr, cc)), shape=(len(p["groups"]), R.shape[0])) @ R)
        .toarray()
        .astype(np.float32)
    )
