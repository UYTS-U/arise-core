# ARISE research code. See NOTICE and CITATION.cff.
"""Independent mixing and strictly lagged endpoint evidence; no tag calibration."""

import numpy as np
import torch
from torch.nn import functional as F
from .data import subset, allowed, public_view
from .memory import read
from .metrics import topk, values


@torch.no_grad()
def static(env, p, hL, hS, tag, window, observed):
    raw = []
    for hist, n in zip(p["history"], p["lengths"]):
        a = (
            np.asarray(env.x[hist[max(0, n - window) : n]].mean(0)).ravel()
            if n
            else np.zeros(env.x.shape[1])
        )
        b = (
            np.asarray(env.x[hist[max(0, n - 50) : max(0, n - window)]].mean(0)).ravel()
            if n > window and window < 50
            else a
        )
        ap = a / max(a.sum(), 1e-08)
        bp = b / max(b.sum(), 1e-08)
        mid = (ap + bp) / 2
        js = 0.5 * np.sum(
            ap * np.log(np.maximum(ap, 1e-12) / np.maximum(mid, 1e-12))
        ) + 0.5 * np.sum(bp * np.log(np.maximum(bp, 1e-12) / np.maximum(mid, 1e-12)))
        raw.append([np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-08), js])
    norm = hL.norm(dim=1).clamp_min(1e-08)
    n = torch.as_tensor(observed, device=hL.device)
    raw = torch.as_tensor(np.array(raw), device=hL.device)
    return (
        torch.stack(
            [
                torch.log1p(n) / np.log(101),
                F.cosine_similarity(hL, hS),
                hS.norm(dim=1) / norm,
                (hS - hL).norm(dim=1) / norm,
                raw[:, 0],
                raw[:, 1],
                tag.norm(dim=1) / norm,
                torch.minimum(n, n.new_tensor(window)) / window,
            ],
            1,
        )
        .float()
        .clamp(-5, 5)
    )


def exclusive_ema(gain, owner, decay=0.8):
    (_, counts) = torch.unique_consecutive(owner, return_counts=True)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    pos = torch.arange(len(gain), device=gain.device) - starts[owner]
    width = int(counts.max())
    pad = torch.zeros(
        (len(counts), width, gain.shape[1]), device=gain.device, dtype=torch.float64
    )
    pad[owner, pos] = gain.double()
    out = torch.zeros_like(pad)
    carry = pad[:, 0] * 0
    for off in range(0, width, 128):
        part = pad[:, off : off + 128]
        j = torch.arange(part.shape[1], device=gain.device, dtype=torch.float64)
        inc = (part * (decay ** (-j))[None, :, None]).cumsum(1)
        exc = torch.cat([inc[:, :1] * 0, inc[:, :-1]], 1)
        out[:, off : off + len(j)] = (
            carry[:, None] * (decay**j)[None, :, None]
            + (1 - decay) * (decay ** (j - 1))[None, :, None] * exc
        )
        carry = carry * decay ** len(j) + (1 - decay) * decay ** (len(j) - 1) * inc[:, -1]
    return out[owner, pos].to(gain.dtype)


@torch.no_grad()
def evidence(long, short, positive, cat, priority):
    lt = topk(long, cat, priority)
    st = topk(short, cat, priority)
    overlap = (lt[:, :, None] == st[:, None, :]).any(2).sum(1).float()
    width = lt.shape[1]
    disagreement = 1 - overlap / (2 * width - overlap).clamp_min(1)
    utility = []
    for scores, top in [(long, lt), (short, st)]:
        metric = values(top, positive)
        lookup = torch.full((len(priority),), -1, device=cat.device, dtype=torch.long)
        lookup[cat] = torch.arange(len(cat), device=cat.device)
        col = lookup[positive]
        ps = scores.gather(1, col.clamp_min(0))
        ranks = torch.zeros_like(ps, dtype=torch.long)
        for z in range(positive.shape[1]):
            ranks[:, z] = (
                (scores > ps[:, z, None])
                | (scores == ps[:, z, None])
                & (priority[cat][None, :] < priority[positive[:, z, None]])
            ).sum(1) + 1
        legal = (col >= 0) & torch.isfinite(ps)
        inv = torch.where(legal, 1 / torch.log2(ranks.float() + 1), torch.zeros_like(ps)).mean(
            1
        )
        utility.append(torch.stack([metric[:, 6].float(), inv, metric[:, 7].float()], 1))
    return (disagreement, utility[1] - utility[0])


def short_history(p, window):
    q = dict(p)
    h = np.zeros_like(p["history"])
    t = np.zeros_like(p["timestamps"])
    lens = np.minimum(p["lengths"], window)
    for j, n in enumerate(p["lengths"]):
        if n:
            h[j, : lens[j]] = p["history"][j, n - lens[j] : n]
            t[j, : lens[j]] = p["timestamps"][j, n - lens[j] : n]
    q.update(history=h, timestamps=t, lengths=lens)
    return q


@torch.no_grad()
def make_frozen(model, env, d, p, condition, window, e, no_cold=False):
    (v, h, tag) = model.user(e, public_view(d), p, condition, no_cold=no_cold)
    sp = short_history(p, window)
    hs = model.history(e, sp["history"], sp["lengths"])
    return dict(
        v=v.detach(),
        hL=h.detach(),
        hS=hs.detach(),
        tag=tag.detach(),
        static=static(env, p, h, hs, tag, window, 5 * d["round_index"][p["groups"]]),
    )


def gate_features(module, env, d, p, b, e, R, scale, li, window, cfg):
    cat = torch.as_tensor(d["catalog"].astype(np.int64), device=e.device)
    priority = torch.as_tensor(env.priority, device=e.device)
    gains = []
    dis = []
    with torch.no_grad():
        for off in range(0, len(p["groups"]), cfg.group_chunk):
            ix = np.arange(off, min(off + cfg.group_chunk, len(p["groups"])))
            q = subset(p, ix)
            c50 = torch.as_tensor(read(q, R, 50), device=e.device)
            cw = torch.as_tensor(read(q, R, window), device=e.device)
            long = b["v"][ix] @ e[cat].T / cfg.tau + li * scale * c50
            first = torch.as_tensor(d["round_index"][q["groups"]] == 0, device=e.device)
            dh = (module.short(b["hS"][ix]) - b["hL"][ix]).masked_fill(first[:, None], 0)
            delta = dh @ e[cat].T / cfg.tau
            short = long + delta + li * scale * (cw - c50)
            ok = allowed(env, d, q)
            pos = torch.as_tensor(d["targets"][q["groups"]].astype(np.int64), device=e.device)
            (disagreement, gain) = evidence(
                long.masked_fill(~ok, -torch.inf),
                short.masked_fill(~ok, -torch.inf),
                pos,
                cat,
                priority,
            )
            dis.append(disagreement)
            gains.append(gain)
        owner = torch.as_tensor(p["owner"], device=e.device)
        ema = exclusive_ema(torch.cat(gains), owner)
        return torch.cat([b["static"], torch.cat(dis)[:, None], ema], 1).clamp(-5, 5)


def score(module, b, features, ix, e, cat, first, c50, cw, li, scale, tau, no_adaptive=False):
    base = b["v"][ix] @ e[cat].T / tau + li * scale * c50
    if no_adaptive:
        return base
    g = module.weights(features[ix], first)
    dh = (module.short(b["hS"][ix]) - b["hL"][ix]).masked_fill(first[:, None], 0)
    delta = dh @ e[cat].T / tau
    return base + g[:, :1] * delta + g[:, 1:] * li * scale * (cw - c50)
