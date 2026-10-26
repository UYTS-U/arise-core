# ARISE research code. See NOTICE and CITATION.cff.
"""Binary unique-target metrics; fixed item priority breaks exact score ties."""

import numpy as np
import torch

NAMES = [m + str(k) for k in [5, 10, 20] for m in ["ndcg", "recall", "precision"]]


def topk(scores, cat, priority, k=20):
    ix = torch.argsort(priority[cat], stable=True)
    order = torch.argsort(scores[:, ix], descending=True, stable=True)[:, : min(k, len(cat))]
    ids = cat[ix[order]]
    return ids.masked_fill(~torch.isfinite(scores.gather(1, ix[order])), -1)


def values(top, positive):
    hit = (top[:, :, None] == positive[:, None, :]).any(2).to(torch.float64)
    valid = ~(positive[:, :, None] == positive[:, None, :]).tril(-1).any(2)
    count = valid.sum(1).clamp_min(1)
    result = []
    for k in [5, 10, 20]:
        n = min(k, top.shape[1])
        discount = 1 / torch.log2(torch.arange(n, device=top.device, dtype=torch.float64) + 2)
        idcg = (torch.arange(n, device=top.device)[None, :] < count[:, None]) * discount
        result.extend(
            [
                (hit[:, :n] * discount).sum(1) / idcg.sum(1).clamp_min(1e-12),
                hit[:, :n].sum(1) / count,
                hit[:, :n].sum(1) / k,
            ]
        )
    return torch.stack(result, 1)


def macro(rows, d, mask=None):
    if mask is None:
        mask = np.ones(len(d["targets"]), bool)
    result = {}
    for part, select in [
        ("first", mask & (d["round_index"] == 0)),
        ("follow", mask & (d["round_index"] > 0)),
    ]:
        per = []
        for u in np.unique(d["round_user"][select]):
            v = rows[select & (d["round_user"] == u)]
            if len(v):
                per.append(np.nanmean(v, axis=0))
        result[part] = dict(
            users=len(per),
            groups=int(select.sum()),
            **dict(
                zip(NAMES, np.nanmean(per, axis=0).tolist() if len(per) else [float("nan")] * 9)
            ),
        )
    return result
