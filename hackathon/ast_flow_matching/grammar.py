"""Tiny arithmetic-expression grammar over fixed-topology binary trees.

A "program" is a perfect binary tree of height ``H`` stored in heap order
(index 0 is the root; the children of node ``i`` live at ``2i+1`` and
``2i+2``). Because the topology is fixed, every array slot has a *known*
type that depends only on its position:

  * internal slots  -> operator tokens  ("+", "-", "*")   (the node has children)
  * leaf slots      -> value tokens      ("x", "1", "2", "3")  (terminal node)

That positional type-map is the "known geometry" of this program space:
we never have to *learn* which slots are operators vs. values, the grammar
tells us up front. A discrete-flow sampler that respects this map can only
ever emit syntactically valid expression trees -- generation becomes
constrained pathfinding over a known mesh rather than blind denoising.

This module defines the vocabulary, the positional type-map, an analytic
"data" distribution, a sampler for it, a validity check, and a pretty
printer. Pure NumPy, no JAX -- it is the problem definition, not the model.
"""

from __future__ import annotations

import numpy as np

# --- vocabulary -------------------------------------------------------------
OPS = ["+", "-", "*"]
VALS = ["x", "1", "2", "3"]
TOKENS = OPS + VALS
V = len(TOKENS)            # number of *real* tokens
MASK = V                   # id of the [MASK] token (model input only)
VOCAB_IN = V + 1           # model input vocabulary (real tokens + MASK)

OP_IDS = list(range(len(OPS)))                          # [0, 1, 2]
VAL_IDS = list(range(len(OPS), len(OPS) + len(VALS)))   # [3, 4, 5, 6]

# Non-uniform "true" generative distribution over each slot's allowed tokens.
# These are the marginals the model has to recover; making them non-uniform
# means the metric measures distribution match, not just validity.
OP_PROBS = np.array([0.5, 0.2, 0.3])         # + - *
VAL_PROBS = np.array([0.4, 0.3, 0.2, 0.1])   # x 1 2 3


def tree_layout(height: int):
    """Return ``(num_nodes, is_leaf)`` for a perfect binary tree of ``height``."""
    n = 2 ** (height + 1) - 1
    first_leaf = 2 ** height - 1
    is_leaf = np.arange(n) >= first_leaf
    return n, is_leaf


def num_nodes(height: int) -> int:
    return 2 ** (height + 1) - 1


def allowed_mask(height: int) -> np.ndarray:
    """``[L, V]`` boolean: which real tokens are grammar-legal at each slot."""
    n, is_leaf = tree_layout(height)
    mask = np.zeros((n, V), dtype=bool)
    for i in range(n):
        mask[i, VAL_IDS if is_leaf[i] else OP_IDS] = True
    return mask


def true_marginals(height: int) -> np.ndarray:
    """``[L, V]`` true per-slot token probabilities under the data process."""
    n, is_leaf = tree_layout(height)
    m = np.zeros((n, V))
    for i in range(n):
        if is_leaf[i]:
            m[i, VAL_IDS] = VAL_PROBS
        else:
            m[i, OP_IDS] = OP_PROBS
    return m


def sample_data(rng: np.random.Generator, batch: int, height: int) -> np.ndarray:
    """Sample ``[batch, L]`` integer token arrays from the data process."""
    n, is_leaf = tree_layout(height)
    out = np.zeros((batch, n), dtype=np.int32)
    for i in range(n):
        if is_leaf[i]:
            out[:, i] = rng.choice(VAL_IDS, size=batch, p=VAL_PROBS)
        else:
            out[:, i] = rng.choice(OP_IDS, size=batch, p=OP_PROBS)
    return out


def is_valid(tokens: np.ndarray, height: int) -> np.ndarray:
    """``tokens``: ``[batch, L]`` ints. Returns ``[batch]`` bool grammar-validity.

    A tree is valid iff every internal slot holds an operator token and every
    leaf slot holds a value token (and no slot holds [MASK]).
    """
    tokens = np.asarray(tokens)
    _, is_leaf = tree_layout(height)
    leaf = is_leaf[None, :]
    is_val_tok = (tokens >= len(OPS)) & (tokens < V)   # value-token ids, not MASK
    is_op_tok = tokens < len(OPS)
    ok = np.where(leaf, is_val_tok, is_op_tok)
    return ok.all(axis=1)


def to_expr(tokens: np.ndarray, height: int, i: int = 0) -> str:
    """Render one tree (``[L]`` ints) as an infix string for display."""
    tokens = np.asarray(tokens)
    _, is_leaf = tree_layout(height)
    tid = int(tokens[i])
    tok = TOKENS[tid] if tid < V else "?"
    if is_leaf[i]:
        return tok
    left = to_expr(tokens, height, 2 * i + 1)
    right = to_expr(tokens, height, 2 * i + 2)
    return f"({left} {tok} {right})"
