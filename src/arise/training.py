# ARISE research code. See NOTICE and CITATION.cff.
"""User-normalized grouped BPR, block freezing, atomic exact epoch resumption."""

import json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import pack, subset, selected_users, allowed, public_view
from .metrics import topk, values, macro
from .memory import read
from .cwm import gate_features, score
from .util import atomic_torch, save_json, progress, rng_state, restore_rng


class Bundle(nn.Module):
    def __init__(self, model, cwm=None):
        super().__init__()
        self.model = model
        if cwm is not None:
            self.cwm = cwm


def cpu_state(module):
    return {k: v.detach().cpu().clone() for (k, v) in module.state_dict().items()}


def set_scope(bundle, scope):
    for p in bundle.parameters():
        p.requires_grad_(False)
    if scope == "theta":
        for name, p in bundle.model.named_parameters():
            p.requires_grad_(not name.startswith("adapters."))
    elif scope == "adapter":
        for p in bundle.model.adapters.parameters():
            p.requires_grad_(True)
    elif scope == "cwm":
        for p in bundle.cwm.parameters():
            p.requires_grad_(True)
    else:
        raise ValueError(scope)


def penalty(bundle, anchor, new_ids):
    terms = []
    count = 0
    for k, p in bundle.named_parameters():
        if not p.requires_grad or k not in anchor:
            continue
        diff = p - anchor[k].to(p.device)
        if k.endswith("ID.weight") and len(new_ids):
            keep = torch.ones(len(diff), dtype=torch.bool, device=p.device)
            keep[new_ids] = False
            diff = diff[keep]
        terms.append(diff.square().sum())
        count += diff.numel()
    return sum(terms) / max(count, 1) if terms else next(bundle.parameters()).sum() * 0


def train_epoch(
    bundle,
    env,
    d,
    mask,
    condition,
    epoch,
    scope,
    opt,
    out,
    R=None,
    li=0.0,
    scale=0.0,
    window=10,
    cache=None,
    anchor=None,
    stable=0.0,
    new_ids=(),
    no_cold=False,
    no_adaptive=False,
):
    cfg = bundle.model.cfg
    model = bundle.model
    model.eval() if scope in ["adapter", "cwm"] else model.train()
    if scope == "adapter":
        model.adapters.train()
    if hasattr(bundle, "cwm"):
        bundle.cwm.train(scope == "cwm")
    order = np.random.default_rng(cfg.seed + epoch).permutation(selected_users(d, mask))
    start = time.time()
    total = 0
    done = 0
    with torch.no_grad():
        frozen_e = model.items().detach() if scope in ["adapter", "cwm"] else None
    for off in range(0, len(order), cfg.user_batch):
        users = order[off : off + cfg.user_batch]
        p = pack(d, users, mask)
        owner = p["owner"]
        counts = np.bincount(owner, minlength=len(users))
        opt.zero_grad(set_to_none=True)
        if scope == "cwm":
            with torch.no_grad():
                e = frozen_e
                b = (cache["clean"] if isinstance(cache, dict) else cache).get(p, e.device)
            features = gate_features(bundle.cwm, env, d, p, b, e, R, scale, li, window, cfg)
        for begin in range(0, len(p["groups"]), cfg.group_chunk):
            ix = np.arange(begin, min(begin + cfg.group_chunk, len(p["groups"])))
            q = subset(p, ix)
            dev = env.device
            pos = torch.as_tensor(d["targets"][q["groups"]].astype(np.int64), device=dev)
            neg = torch.as_tensor(d["negatives"][q["groups"]].astype(np.int64), device=dev)
            if scope == "cwm":
                cat = torch.cat([pos, neg], 1)
                globalcat = d["catalog"]
                lookup = np.full(env.nitems, -1, np.int64)
                lookup[globalcat] = np.arange(len(globalcat))
                c50 = read(q, R, 50)
                cw = read(q, R, window)
                cols = lookup[cat.cpu().numpy()]
                assert cols.min() >= 0
                a = torch.as_tensor(np.take_along_axis(c50, cols, axis=1), device=dev)
                z = torch.as_tensor(np.take_along_axis(cw, cols, axis=1), device=dev)
                base = (b["v"][ix, None] * e[cat]).sum(-1) / cfg.tau + li * scale * a
                first = torch.as_tensor(d["round_index"][q["groups"]] == 0, device=dev)
                g = bundle.cwm.weights(features[ix], first)
                delta = ((bundle.cwm.short(b["hS"][ix]) - b["hL"][ix])[:, None] * e[cat]).sum(
                    -1
                ) / cfg.tau
                scores = base + g[:, :1] * delta + g[:, 1:] * li * scale * (z - a)
                sp = scores[:, :5]
                sn = scores[:, 5:]
            else:
                e = frozen_e if frozen_e is not None else model.items()
                (v, h, tag) = model.user(
                    e,
                    public_view(d),
                    q,
                    condition,
                    cold_view=scope == "adapter",
                    no_cold=no_cold or scope == "theta",
                )
                sp = (v[:, None] * e[pos]).sum(-1) / cfg.tau
                sn = (v[:, None] * e[neg]).sum(-1) / cfg.tau
            per = F.softplus(sn[:, :, None] - sp[:, None, :]).mean((1, 2))
            den = torch.as_tensor(counts[q["owner"]], device=dev)
            loss = (per / den).sum() / len(users)
            if loss.requires_grad:
                loss.backward()
            total += float(loss.detach()) * len(users)
        if anchor is not None and stable:
            reg = stable * penalty(bundle, anchor, new_ids)
            reg.backward()
        torch.nn.utils.clip_grad_norm_([p for p in bundle.parameters() if p.requires_grad], 5)
        for p0 in bundle.parameters():
            if p0.grad is not None and (not torch.isfinite(p0.grad).all()):
                raise FloatingPointError("Nonfinite gradients")
        opt.step()
        done += len(users)
        if off // cfg.user_batch % 10 == 0:
            progress(
                out,
                "training " + scope,
                stage=d["stage"],
                epoch=epoch,
                users=done,
                total_users=len(order),
                seconds=time.time() - start,
            )
    torch.cuda.synchronize()
    return dict(
        loss=total / max(done, 1),
        seconds=time.time() - start,
        users=done,
        groups=int(mask.sum()),
    )


@torch.no_grad()
def evaluate(
    bundle,
    env,
    d,
    mask,
    condition,
    out,
    scope="final",
    R=None,
    li=0.0,
    scale=0.0,
    window=10,
    no_cold=False,
    no_adaptive=False,
    cache=None,
):
    model = bundle.model
    cfg = model.cfg
    bundle.eval()
    e = model.items().detach()
    cat = torch.as_tensor(d["catalog"].astype(np.int64), device=e.device)
    priority = torch.as_tensor(env.priority, device=e.device)
    result = np.full((len(d["targets"]), 9), np.nan, np.float32)
    users = selected_users(d, mask)
    for off in range(0, len(users), cfg.user_batch):
        p = pack(d, users[off : off + cfg.user_batch])
        if scope in ["cwm", "final"] and hasattr(bundle, "cwm") and (not no_adaptive):
            b = (cache[condition] if isinstance(cache, dict) else cache).get(p, e.device)
            features = gate_features(bundle.cwm, env, d, p, b, e, R, scale, li, window, cfg)
        for begin in range(0, len(p["groups"]), cfg.group_chunk):
            ix = np.arange(begin, min(begin + cfg.group_chunk, len(p["groups"])))
            ix = ix[mask[p["groups"][ix]]]
            if not len(ix):
                continue
            q = subset(p, ix)
            if scope in ["cwm", "final"] and hasattr(bundle, "cwm") and (not no_adaptive):
                c50 = torch.as_tensor(read(q, R, 50), device=e.device)
                cw = torch.as_tensor(read(q, R, window), device=e.device)
                first = torch.as_tensor(d["round_index"][q["groups"]] == 0, device=e.device)
                logits = score(
                    bundle.cwm, b, features, ix, e, cat, first, c50, cw, li, scale, cfg.tau
                )
            else:
                (v, h, tag) = model.user(
                    e,
                    public_view(d),
                    q,
                    condition,
                    cold_view=scope == "adapter",
                    no_cold=no_cold or scope == "theta",
                )
                logits = v @ e[cat].T / cfg.tau
                if R is not None:
                    logits += li * scale * torch.as_tensor(read(q, R, 50), device=e.device)
            logits.masked_fill_(~allowed(env, d, q), -torch.inf)
            top = topk(logits, cat, priority)
            pos = torch.as_tensor(d["targets"][q["groups"]].astype(np.int64), device=e.device)
            result[q["groups"]] = values(top, pos).cpu().numpy()
        progress(
            out,
            "validation" if not mask.all() else "evaluation",
            stage=d["stage"],
            condition=condition,
            users=min(off + cfg.user_batch, len(users)),
            total_users=len(users),
        )
    assert np.isfinite(result[mask]).all()
    return result


def objective(bundle, env, d, mask, out, scope, kwargs):
    parts = {}
    conditions = ["clean", "deleted60"] if scope == "cwm" else ["clean", "replaced60"]
    for c in conditions:
        v = evaluate(bundle, env, d, mask, c, out, scope=scope, **kwargs)
        parts[c] = macro(v, d, mask)
    if scope == "adapter":
        return (
            0.5 * parts["clean"]["first"]["ndcg5"]
            + 0.25 * parts["clean"]["first"]["ndcg10"]
            + 0.25 * parts["replaced60"]["first"]["ndcg10"]
        )
    if scope == "cwm":
        return np.mean([parts[c]["follow"][m] for c in conditions for m in ["ndcg5", "ndcg10"]])
    return np.mean([parts[c][p]["ndcg10"] for c in conditions for p in ["first", "follow"]])


def fit(bundle, env, d, mask, out, name, scope, maximum, validation=True, **kwargs):
    path = Path(out) / "training" / name
    path.mkdir(parents=True, exist_ok=True)
    done = path / "done.json"
    cfg = bundle.model.cfg
    set_scope(bundle, scope)
    if done.exists():
        state = torch.load(path / "best.pt", map_location="cpu")
        bundle.load_state_dict(state["model"])
        restore_rng(state["rng"])
        return json.loads(done.read_text())
    opt = torch.optim.AdamW(
        [p for p in bundle.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    latest = path / "latest.pt"
    start = 0
    best = -np.inf
    bad = 0
    best_epoch = 0
    curve = []
    if latest.exists():
        s = torch.load(latest, map_location="cpu")
        bundle.load_state_dict(s["model"])
        opt.load_state_dict(s["optimizer"])
        restore_rng(s["rng"])
        start = s["epoch"]
        best = s["best"]
        bad = s["bad"]
        best_epoch = s["best_epoch"]
        curve = s["curve"]
        if best_epoch == start and (
            not (path / "best.pt").exists()
            or torch.load(path / "best.pt", map_location="cpu")["epoch"] != best_epoch
        ):
            atomic_torch(path / "best.pt", s)
    validation_kwargs = {
        k: v
        for (k, v) in kwargs.items()
        if k in ["R", "li", "scale", "window", "no_cold", "no_adaptive", "cache"]
    }
    epoch = start
    stop_already = validation and start >= cfg.minimum_epochs and (bad >= cfg.patience)
    for epoch in range(start + 1, start + 1 if stop_already else maximum + 1):
        eligible = True
        train = train_epoch(bundle, env, d, mask, "clean", epoch, scope, opt, out, **kwargs)
        j = (
            float(objective(bundle, env, d, d["validation"], out, scope, validation_kwargs))
            if validation
            else -train["loss"]
        )
        if not np.isfinite(j):
            raise ValueError("Undefined validation objective")
        curve.append(dict(epoch=epoch, objective=j, selection_eligible=eligible, **train))
        improved = j > best and eligible
        if improved:
            best = j
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
        state = dict(
            model=cpu_state(bundle),
            optimizer=opt.state_dict(),
            rng=rng_state(),
            epoch=epoch,
            best=best,
            bad=bad,
            best_epoch=best_epoch,
            curve=curve,
        )
        atomic_torch(latest, state)
        if improved or not validation:
            atomic_torch(path / "best.pt", state)
        save_json(path / "curve.json", curve)
        progress(
            out,
            "epoch complete",
            block=name,
            epoch=epoch,
            best_epoch=best_epoch,
            objective=j,
            seconds=train["seconds"],
        )
        if validation and eligible and (epoch >= cfg.minimum_epochs) and (bad >= cfg.patience):
            break
    s = torch.load(path / "best.pt", map_location="cpu")
    bundle.load_state_dict(s["model"])
    restore_rng(s["rng"])
    result = dict(
        actual_epochs=epoch,
        selected_epoch=best_epoch if validation else epoch,
        selected_objective=best,
        maximum=maximum,
        scope=scope,
        validation=validation,
        trainable_parameters=sum((p.numel() for p in bundle.parameters() if p.requires_grad)),
        trainable_names=[k for (k, p) in bundle.named_parameters() if p.requires_grad],
    )
    save_json(done, result)
    return result
