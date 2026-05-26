"""Masked (absorbing) discrete flow matching over the op tokens of a DAG.

Generation is *conditioned on a scaffold*: the node arities and the edge
structure are given (the known geometry / the "mesh"), and the flow fills in the
operation at each node. The source state is all-[MASK]; the absorbing process
reveals nodes over time ``t: 0 -> 1`` at rate ``dt / (1 - t)``, drawing each
revealed op from the model's predicted ``p(op | partial program, scaffold)``.

This is identical machinery to the AST prototype; the only change is that the
denoiser is a graph Transformer reading data-/control-flow edges, not a flat
sequence model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from data import MASK
from model import compute_relations, legal_bias


def make_batch(op, node_type, parents, legal_mask, device):
    """NumPy arrays -> device tensors + precomputed R and legal logit-bias."""
    op = torch.as_tensor(op, device=device)
    node_type = torch.as_tensor(node_type, device=device)
    parents = torch.as_tensor(parents, device=device)
    R = compute_relations(parents)
    lbias = legal_bias(node_type, legal_mask)
    return {"op": op, "node_type": node_type, "parents": parents,
            "R": R, "lbias": lbias}


def corrupt(op, t, mask_id=MASK):
    keep = torch.rand(op.shape, device=op.device) < t[:, None]
    return torch.where(keep, op, torch.full_like(op, mask_id))


def loss(model, batch):
    op = batch["op"]
    B = op.shape[0]
    t = torch.rand(B, device=op.device).clamp_(1e-3, 1.0)
    xt = corrupt(op, t)
    logits = model(xt, t, batch["node_type"], batch["R"], batch["lbias"])
    logp = F.log_softmax(logits, dim=-1)
    tgt = torch.gather(logp, -1, op[..., None])[..., 0]      # [B, N]
    masked = (xt == MASK).float()
    return -(tgt * masked).sum() / masked.sum().clamp(min=1.0)


def _parents_ready(parents, revealed):
    """``[B, N]`` bool: every parent of each node is already revealed."""
    B, N, A = parents.shape
    pad = parents < 0
    idx = parents.clamp(min=0).reshape(B, N * A)
    g = torch.gather(revealed.float(), 1, idx).reshape(B, N, A)
    g = torch.where(pad, torch.ones_like(g), g)
    return g.bool().all(dim=-1)


@torch.no_grad()
def sample(model, batch, steps=64, topo=False):
    """Fill ops on the batch's scaffold. Returns generated op tokens ``[B, N]``.

    With ``topo=True`` the known DAG is used to unmask in dependency order: a node
    only becomes eligible once all its parents are revealed, so the denoiser
    always sees concrete parent ops and can condition on the data flow. This is
    the graph-guided ("pathfinding over the mesh") decoder; it needs the edges,
    so only the full-geometry model can use it.
    """
    op = batch["op"]
    x = torch.full_like(op, MASK)
    parents = batch["parents"]
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        tt = torch.full((x.shape[0],), t, device=x.device)
        logits = model(x, tt, batch["node_type"], batch["R"], batch["lbias"])
        proposal = torch.distributions.Categorical(logits=logits).sample()
        is_masked = x == MASK
        cand = is_masked
        if topo:
            cand = cand & _parents_ready(parents, ~is_masked)
        unmask_prob = min(dt / (1.0 - t), 1.0)
        do = cand & (torch.rand(x.shape, device=x.device) < unmask_prob)
        x = torch.where(do, proposal, x)

    # finalize any leftovers in dependency order (deterministic)
    for _ in range(x.shape[1]):
        is_masked = x == MASK
        if not is_masked.any():
            break
        cand = is_masked
        if topo:
            cand = cand & _parents_ready(parents, ~is_masked)
        tt = torch.full((x.shape[0],), 1.0, device=x.device)
        logits = model(x, tt, batch["node_type"], batch["R"], batch["lbias"])
        x = torch.where(cand, logits.argmax(-1), x)
    return x
