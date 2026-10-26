# ARISE research code. See NOTICE and CITATION.cff.
"""Order is data semantics. Feedback never changes the declared-tag weights."""

import torch


def weights(tags, ordered, pi0=0.3, gamma=0.35):
    valid = tags >= 0
    logits = torch.zeros_like(tags, dtype=torch.float32)
    if ordered:
        logits = logits - gamma * torch.arange(tags.shape[1], device=tags.device)[None, :]
    logits = logits.masked_fill(~valid, -torch.inf)
    empty = ~valid.any(1)
    logits[empty] = 0
    w = torch.softmax(logits, 1) * valid * (1 - pi0)
    ref = torch.where(empty, w.new_ones((len(tags),)), w.new_full((len(tags),), pi0))
    return (ref, w)
