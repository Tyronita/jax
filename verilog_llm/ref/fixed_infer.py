"""Bit-accurate fixed-point forward pass.

This mirrors, op-for-op, what rtl/gpt_core.v computes.  It consumes the
quantized integer weights and produces integer logits / generated tokens that
the Verilog must reproduce exactly.
"""
import fixed as F

def matvec(W, x, bias, n_in, n_out):
    """y[o] = sum_k W[o,k]*x[k]  (+bias).  W stored row-major, Q*Q>>FRAC."""
    y = [0]*n_out
    for o in range(n_out):
        acc = 0
        base = o*n_in
        for k in range(n_in):
            acc += W[base+k]*x[k]          # full-precision int accumulate
        v = F.sat(acc >> F.FRAC)
        if bias is not None:
            v = F.sat(v + bias[o])
        y[o] = v
    return y

def layernorm(x, g, b, D, eps_q):
    s = 0
    for i in range(D): s += x[i]
    mean = F.tdiv(s, D) if hasattr(F,'tdiv') else trunc_div(s, D)
    xc = [F.sat(x[i]-mean) for i in range(D)]
    sv = 0
    for i in range(D): sv += F.fmul(xc[i], xc[i])
    var = trunc_div(sv, D)
    inv = F.frsqrt(F.sat(var + eps_q))
    out = [0]*D
    for i in range(D):
        xn = F.fmul(xc[i], inv)
        out[i] = F.sat(F.fmul(xn, g[i]) + b[i])
    return out

def trunc_div(a, n):
    q = abs(a)//abs(n)
    if (a<0)^(n<0): q=-q
    return q

def softmax_row(sc, valid, INV_SQRT_D):
    # sc already scaled? we scale here
    s = [F.fmul(sc[j], INV_SQRT_D) for j in range(valid)]
    m = s[0]
    for j in range(1, valid):
        if s[j] > m: m = s[j]
    e = [F.fexp_neg(F.sat(s[j]-m)) for j in range(valid)]
    tot = 0
    for j in range(valid): tot += e[j]
    if tot == 0: tot = 1
    return [F.fdiv(e[j], tot) for j in range(valid)]

class FixedGPT:
    def __init__(self, q, cfg, consts):
        self.q = q; self.cfg = cfg
        self.INV_SQRT_D = consts['INV_SQRT_D']; self.EPS = consts['EPS']

    def forward(self, seq):
        q = self.q; D=self.cfg['D']; M=self.cfg['M']; L=self.cfg['L']; T=len(seq)
        # embeddings
        x = [[F.sat(q['tok'][seq[t]*D+i] + q['pos'][t*D+i]) for i in range(D)] for t in range(T)]
        for l in range(L):
            h = [layernorm(x[t], q[f'ln1g{l}'], q[f'ln1b{l}'], D, self.EPS) for t in range(T)]
            qv=[matvec(q[f'Wq{l}'],h[t],q[f'bq{l}'],D,D) for t in range(T)]
            kv=[matvec(q[f'Wk{l}'],h[t],q[f'bk{l}'],D,D) for t in range(T)]
            vv=[matvec(q[f'Wv{l}'],h[t],q[f'bv{l}'],D,D) for t in range(T)]
            o=[[0]*D for _ in range(T)]
            for t in range(T):
                sc=[0]*(t+1)
                for j in range(t+1):
                    acc=0
                    for d in range(D): acc += qv[t][d]*kv[j][d]
                    sc[j]=F.sat(acc>>F.FRAC)
                a=softmax_row(sc, t+1, self.INV_SQRT_D)
                for d in range(D):
                    acc=0
                    for j in range(t+1): acc += a[j]*vv[j][d]
                    o[t][d]=F.sat(acc>>F.FRAC)
            op=[matvec(q[f'Wo{l}'],o[t],q[f'bo{l}'],D,D) for t in range(T)]
            x=[[F.sat(x[t][i]+op[t][i]) for i in range(D)] for t in range(T)]
            h2=[layernorm(x[t], q[f'ln2g{l}'], q[f'ln2b{l}'], D, self.EPS) for t in range(T)]
            for t in range(T):
                z1=matvec(q[f'W1{l}'],h2[t],q[f'b1{l}'],D,M)
                a1=[v if v>0 else 0 for v in z1]
                m=matvec(q[f'W2{l}'],a1,q[f'b2{l}'],M,D)
                for i in range(D): x[t][i]=F.sat(x[t][i]+m[i])
        xf=[layernorm(x[t], q['lnfg'], q['lnfb'], D, self.EPS) for t in range(T)]
        V=self.cfg['V']
        logits=[matvec(q['Wh'],xf[t],q['bh'],D,V) for t in range(T)]
        return logits

    def generate(self, prompt, T):
        seq = list(prompt) + [0]*(T-len(prompt))
        cur = len(prompt)
        out = list(prompt)
        while cur < T:
            logits = self.forward(seq)
            row = logits[cur-1]
            nxt = max(range(len(row)), key=lambda i: row[i])
            seq[cur] = nxt
            out.append(nxt)
            cur += 1
        return out
