# ARISE research code. See NOTICE and CITATION.cff.
import math
import numpy as np
import torch
from torch import nn
from ..tags import weights
from ..data import tag_arrays


class TagLaunch(nn.Sequential):
    def __init__(self, d):
        super().__init__(
            nn.Linear(d, 2 * d, bias=False), nn.Tanh(), nn.Linear(2 * d, d, bias=False)
        )
        nn.init.xavier_uniform_(self[0].weight)
        nn.init.zeros_(self[2].weight)


class ARISE(nn.Module):
    def __init__(self, env, cfg):
        super().__init__()
        self.env = env
        self.cfg = cfg
        d = cfg.dim
        self.Q = nn.Parameter(torch.randn(env.x.shape[1], d) * 0.02)
        self.M = nn.ParameterList(
            [nn.Parameter(torch.randn(n, d) * 0.02) for n in env.spec["tag_sizes"]]
        )
        self.Wg = nn.Linear(d, d, bias=False)
        self.Wt = nn.Linear(env.text.shape[1], d, bias=False)
        nn.init.xavier_uniform_(self.Wg.weight)
        nn.init.xavier_uniform_(self.Wt.weight)
        self.ID = nn.Embedding(env.nitems, d)
        nn.init.zeros_(self.ID.weight)
        self.register_buffer("rho", torch.zeros(env.nitems, dtype=torch.bool))
        self.position = nn.Embedding(cfg.max_history, d)
        nn.init.normal_(self.position.weight, std=0.02)
        self.h_empty = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d,
            cfg.heads,
            4 * d,
            cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, cfg.layers, norm=nn.LayerNorm(d), enable_nested_tensor=False
        )
        self.adapters = nn.ModuleList([TagLaunch(d) for _ in self.M])
        self.address_projection = (
            nn.Linear(6, d, bias=False) if env.address_features is not None else None
        )
        self.register_buffer(
            "address_seen",
            torch.zeros(env.spec["tag_sizes"][-1], dtype=torch.bool)
            if self.address_projection
            else torch.zeros(0, dtype=torch.bool),
        )

    def enable(self, ids, d=None, mask=None):
        self.rho[torch.as_tensor(ids, device=self.rho.device)] = True
        if self.address_projection and d is not None:
            users = np.unique(d["round_user"][mask])
            a = d["address"][users]
            a = a[a >= 0]
            self.address_seen[torch.as_tensor(a.astype(np.int64), device=self.rho.device)] = (
                True
            )

    def items(self):
        return (
            self.Wg(self.env.project(self.Q))
            + self.Wt(self.env.text)
            + self.rho[:, None] * self.ID.weight
        )

    def history(self, items, history, lengths):
        dev = items.device
        history = torch.as_tensor(history, device=dev)
        lengths = torch.as_tensor(lengths, device=dev)
        out = self.h_empty.expand(len(lengths), -1).clone()
        valid = lengths > 0
        if valid.any():
            lens = lengths[valid]
            width = int(lens.max())
            ids = history[valid, :width]
            pos = torch.arange(width, device=dev)
            x = items[ids] + self.position(pos)
            padding = pos[None] >= lens[:, None]
            x = self.encoder(
                x,
                mask=torch.ones(width, width, device=dev, dtype=torch.bool).triu(1),
                src_key_padding_mask=padding,
            )
            out[valid] = x[torch.arange(len(lens), device=dev), lens - 1]
        return out

    def tag_states(self, d, p, condition):
        out = []
        for q, a in enumerate(tag_arrays(d, condition)):
            tags = torch.as_tensor(a[p["users"]].astype(np.int64), device=self.Q.device)
            (ref, w) = weights(
                tags, self.env.spec["tag_order"][q], self.cfg.pi0, self.cfg.gamma
            )
            proto = self.M[q]
            if q == 1 and self.address_projection:
                proto = proto * self.address_seen[:, None] + self.address_projection(
                    self.env.address_features
                )
            out.append(
                (w[:, :, None] * proto[tags.clamp_min(0)]).sum(1)[
                    torch.as_tensor(p["owner"], device=self.Q.device)
                ]
            )
        return out

    def user(self, e, d, p, condition="clean", cold_view=False, no_cold=False):
        states = self.tag_states(d, p, condition)
        tag = sum(states)
        h = (
            self.h_empty.expand(len(tag), -1)
            if cold_view
            else self.history(e, p["history"], p["lengths"])
        )
        first = torch.as_tensor(d["round_index"][p["groups"]] == 0, device=e.device)
        v = h + tag
        if not no_cold:
            v = (
                v
                + sum((a(m) for (a, m) in zip(self.adapters, states)))
                * (torch.ones_like(first) if cold_view else first)[:, None]
            )
        return (v, h, tag)


class CoupledWindowMixer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.down = nn.Linear(cfg.dim, cfg.rank, bias=False)
        self.up = nn.Linear(cfg.rank, cfg.dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(12, 16), nn.Tanh(), nn.Linear(16, 2))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.xavier_uniform_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
        nn.init.zeros_(self.gate[2].weight)
        nn.init.constant_(self.gate[2].bias, math.log(0.1 / 0.9))

    def short(self, h):
        return h + self.up(self.down(h))

    def weights(self, x, first):
        return torch.sigmoid(self.gate(x)).masked_fill(first[:, None], 0)
