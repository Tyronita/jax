"""A tiny imperative DSL whose programs are computation DAGs.

Unlike pure arithmetic expressions (which only have an AST), these programs
carry genuine **data-flow** and **control-flow** structure:

  * data flow  -- every non-leaf node consumes the values produced by earlier
                  nodes (its *parents*); those edges are the data-flow graph.
  * control flow -- a ``SELECT(cond, a, b)`` node picks one of two inputs based
                  on a predicate (the SSA / phi-function encoding of a branch),
                  so the cond-edge is a control dependency.

A program is a fixed-size, topologically-ordered DAG of ``N`` nodes:

  * the first few nodes are leaves  (in-degree 0)  -> INPUT or a constant
  * every later node is binary (in-degree 2) or ternary/SELECT (in-degree 3),
    wired to distinct earlier nodes.

The *scaffold* of a program -- node arities + the wiring (with operand roles) --
is the "known geometry". It is computable, it constrains which op is legal at
each node, and it tells you exactly which earlier nodes each node depends on.
The generative task is: given a scaffold, fill in the operations. A model that
respects the scaffold emits only valid, runnable programs and can read the data
flow directly; a flat sequence model has to rediscover all of it.

Pure NumPy -- this is the problem definition, no torch here.
"""

from __future__ import annotations

import numpy as np

# --- op vocabulary ----------------------------------------------------------
OPS = ["INPUT", "C0", "C1", "C2", "ADD", "SUB", "MUL", "LT", "EQ", "SELECT"]
V = len(OPS)               # number of real ops
MASK = V                   # [MASK] token (model input only)
VOCAB_IN = V + 1

OP2ID = {o: i for i, o in enumerate(OPS)}

# node arity classes and the ops legal at each
LEAF, BINARY, TERNARY = 0, 1, 2
ARITY = {LEAF: 0, BINARY: 2, TERNARY: 3}
MAX_ARITY = 3
LEGAL_OPS = {
    LEAF: [OP2ID[o] for o in ("INPUT", "C0", "C1", "C2")],
    BINARY: [OP2ID[o] for o in ("ADD", "SUB", "MUL", "LT", "EQ")],
    TERNARY: [OP2ID["SELECT"]],
}
N_REL = 6   # relation types: 0 none, 1 self, 2/3/4 parent-arg0/1/2, 5 child


def legal_mask() -> np.ndarray:
    """``[3, V]`` boolean: which ops are grammar-legal for each node arity."""
    m = np.zeros((3, V), dtype=bool)
    for t, ops in LEGAL_OPS.items():
        m[t, ops] = True
    return m


# --- the "true" data-generating distribution --------------------------------
# Leaf ops come from a fixed non-uniform marginal. The interesting part: a
# BINARY node's op is drawn conditioned on *its two parents' ops*, so the signal
# lives on the data-flow edges. A model that can see the parents can learn it; a
# flat sequence model (which doesn't know which earlier nodes are the parents)
# cannot do better than the marginal.
def make_distribution(seed: int = 0):
    rng = np.random.default_rng(seed)
    leaf_marg = rng.dirichlet(np.ones(len(LEGAL_OPS[LEAF])) * 0.7)
    # cond[p0, p1] -> distribution over the 5 binary ops; low-entropy (alpha<1).
    cond = rng.dirichlet(np.ones(len(LEGAL_OPS[BINARY])) * 0.3, size=(V, V))
    return {"leaf_marg": leaf_marg, "cond": cond}


def sample_program(rng, N, dist, p_ternary=0.25):
    """Sample one program. Returns op[N], node_type[N], parents[N, 3] (-1 pad)."""
    n_leaves = max(2, N // 4)
    op = np.full(N, -1, dtype=np.int64)
    node_type = np.zeros(N, dtype=np.int64)
    parents = np.full((N, MAX_ARITY), -1, dtype=np.int64)

    leaf_ops = LEGAL_OPS[LEAF]
    bin_ops = np.array(LEGAL_OPS[BINARY])
    for i in range(N):
        if i < n_leaves:
            node_type[i] = LEAF
            op[i] = rng.choice(leaf_ops, p=dist["leaf_marg"])
            continue
        can_ternary = i >= 3 and rng.random() < p_ternary
        if can_ternary:
            node_type[i] = TERNARY
            ps = rng.choice(i, size=3, replace=False)
            parents[i, :3] = ps
            op[i] = OP2ID["SELECT"]
        else:
            node_type[i] = BINARY
            ps = rng.choice(i, size=2, replace=False)
            parents[i, :2] = ps
            p0, p1 = int(op[ps[0]]), int(op[ps[1]])
            op[i] = rng.choice(bin_ops, p=dist["cond"][p0, p1])
    return op, node_type, parents


def sample_dataset(seed, n, N, dist, p_ternary=0.25):
    """Generate ``n`` programs as stacked arrays."""
    rng = np.random.default_rng(seed)
    ops = np.empty((n, N), dtype=np.int64)
    types = np.empty((n, N), dtype=np.int64)
    pars = np.empty((n, N, MAX_ARITY), dtype=np.int64)
    for k in range(n):
        ops[k], types[k], pars[k] = sample_program(rng, N, dist, p_ternary)
    return ops, types, pars


# --- structural helpers -----------------------------------------------------
def is_valid(op, node_type) -> np.ndarray:
    """``[B]`` bool: every node's op is legal for its arity (and not MASK)."""
    op = np.asarray(op)
    node_type = np.asarray(node_type)
    lm = legal_mask()                       # [3, V]
    safe = np.clip(op, 0, V - 1)
    legal_here = np.take_along_axis(lm[node_type], safe[..., None], axis=-1)[..., 0]
    in_range = (op >= 0) & (op < V)
    return (legal_here & in_range).all(axis=-1)


def execute(op, parents, node_type, inputs):
    """Evaluate one program on a list of input ints; returns the output value.

    Raises if the program is structurally malformed (used to confirm that
    structurally valid programs are also runnable).
    """
    N = len(op)
    val = [0] * N
    inp_iter = iter(inputs)
    for i in range(N):
        o = OPS[op[i]]
        if o == "INPUT":
            val[i] = next(inp_iter, 0)
        elif o == "C0":
            val[i] = 0
        elif o == "C1":
            val[i] = 1
        elif o == "C2":
            val[i] = 2
        elif o in ("ADD", "SUB", "MUL", "LT", "EQ"):
            a, b = val[parents[i, 0]], val[parents[i, 1]]
            val[i] = {"ADD": a + b, "SUB": a - b, "MUL": a * b,
                      "LT": int(a < b), "EQ": int(a == b)}[o]
        elif o == "SELECT":
            c, a, b = val[parents[i, 0]], val[parents[i, 1]], val[parents[i, 2]]
            val[i] = a if c != 0 else b
        else:
            raise ValueError(f"node {i}: op {o} illegal for type {node_type[i]}")
    return val[N - 1]


def to_code(op, parents, node_type) -> str:
    """Render one program as SSA-style pseudocode."""
    lines = []
    n_inputs = 0
    for i in range(len(op)):
        o = OPS[op[i]] if 0 <= op[i] < V else "?"
        if o == "INPUT":
            lines.append(f"n{i} = input[{n_inputs}]")
            n_inputs += 1
        elif o in ("C0", "C1", "C2"):
            lines.append(f"n{i} = {o[1]}")
        elif o == "SELECT":
            p = parents[i]
            lines.append(f"n{i} = n{p[1]} if n{p[0]} else n{p[2]}")
        elif o in ("ADD", "SUB", "MUL", "LT", "EQ"):
            sym = {"ADD": "+", "SUB": "-", "MUL": "*", "LT": "<", "EQ": "=="}[o]
            p = parents[i]
            lines.append(f"n{i} = n{p[0]} {sym} n{p[1]}")
        else:
            lines.append(f"n{i} = ?op{op[i]}  (illegal for {['leaf','bin','tern'][node_type[i]]})")
    return "\n".join(lines)
