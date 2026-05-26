"""Relational graph Transformer (PyTorch), plus the structural inputs.

All three ablation variants are the *same* module with three switches:

  * ``use_type`` : add a node-arity embedding (the model knows leaf/binary/ternary)
  * ``use_rel``  : add a per-head attention bias indexed by the edge relation
                   between every pair of nodes (self / parent-arg0/1/2 / child) --
                   this is how the data-/control-flow graph enters the model
  * ``use_mask`` : mask output logits to the ops legal for each node's arity

So the variants are:

  seq        use_type=F use_rel=F use_mask=F   "throw away the graph" baseline
  seq+mask   use_type=T use_rel=F use_mask=T   knows arities (valid) but not edges
  graph      use_type=T use_rel=T use_mask=T   full known geometry

Attention is full (every node sees every node) in all three; the graph variant
just *adds* the edge bias, so the comparison isolates exactly what the structure
buys you rather than confounding it with a different attention pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import V, VOCAB_IN, N_REL, MAX_ARITY


@dataclass
class Config:
    N: int
    d: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    use_type: bool = True
    use_rel: bool = True
    use_mask: bool = True


def compute_relations(parents: torch.Tensor) -> torch.Tensor:
    """``parents`` ``[B, N, MAX_ARITY]`` (-1 pad) -> relation matrix ``R`` ``[B, N, N]``.

    ``R[b, i, j]`` is the relation of key ``j`` w.r.t. query ``i``:
    1=self, 2/3/4 = j is parent of i in arg slot 0/1/2, 5 = j is a child of i,
    0 = unrelated.
    """
    B, N, A = parents.shape
    dev = parents.device
    R = torch.zeros(B, N, N, dtype=torch.long, device=dev)
    bb = torch.arange(B, device=dev).view(B, 1).expand(B, N)
    ii = torch.arange(N, device=dev).view(1, N).expand(B, N)
    for r in range(A):
        p = parents[:, :, r]                 # [B, N]
        m = p >= 0
        if m.any():
            b_sel, i_sel, p_sel = bb[m], ii[m], p[m]
            R[b_sel, i_sel, p_sel] = 2 + r   # key p is parent (arg r) of query i
            R[b_sel, p_sel, i_sel] = 5        # key i is a child of query p
    eye = torch.arange(N, device=dev)
    R[:, eye, eye] = 1
    return R


def legal_bias(node_type: torch.Tensor, legal_mask: torch.Tensor, neg=-1e9):
    """``node_type`` ``[B, N]``, ``legal_mask`` ``[3, V]`` bool -> ``[B, N, V]`` bias."""
    lm = legal_mask[node_type]               # [B, N, V] bool
    return torch.where(lm, torch.zeros((), device=lm.device), torch.full((), neg, device=lm.device))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_heads
        self.dh = cfg.d // cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d)
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d)
        self.out = nn.Linear(cfg.d, cfg.d)
        self.ln2 = nn.LayerNorm(cfg.d)
        self.ff = nn.Sequential(nn.Linear(cfg.d, cfg.d_ff), nn.GELU(),
                                nn.Linear(cfg.d_ff, cfg.d))

    def forward(self, x, rel_bias):
        B, N, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # [B, H, N, dh]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)  # [B, H, N, N]
        if rel_bias is not None:
            att = att + rel_bias
        att = F.softmax(att, dim=-1)
        o = (att @ v).transpose(1, 2).reshape(B, N, d)
        x = x + self.out(o)
        x = x + self.ff(self.ln2(x))
        return x


class GraphTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.op_emb = nn.Embedding(VOCAB_IN, cfg.d)
        self.pos_emb = nn.Embedding(cfg.N, cfg.d)
        self.type_emb = nn.Embedding(3, cfg.d)
        self.rel_emb = nn.Embedding(N_REL, cfg.n_heads)     # per-head edge bias
        self.time_mlp = nn.Sequential(nn.Linear(1, cfg.d), nn.SiLU(),
                                      nn.Linear(cfg.d, cfg.d))
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d)
        self.head = nn.Linear(cfg.d, V)
        nn.init.zeros_(self.rel_emb.weight)

    def forward(self, op, t, node_type, R, lbias):
        B, N = op.shape
        pos = torch.arange(N, device=op.device)
        x = self.op_emb(op) + self.pos_emb(pos)[None]
        if self.cfg.use_type:
            x = x + self.type_emb(node_type)
        x = x + self.time_mlp(t[:, None])[:, None, :]
        rel_bias = None
        if self.cfg.use_rel:
            rel_bias = self.rel_emb(R).permute(0, 3, 1, 2)   # [B, H, N, N]
        for blk in self.blocks:
            x = blk(x, rel_bias)
        logits = self.head(self.ln_f(x))                     # [B, N, V]
        if self.cfg.use_mask:
            logits = logits + lbias
        return logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
