"""Tiny GPT (decoder-only transformer), float32, with manual backprop.

Architecture (pre-norm), single attention head:
  x = tok_emb[t] + pos_emb
  per layer:  x += attn(LN(x));  x += mlp(LN(x))
  logits = head(LN_f(x))
"""
import numpy as np

class Config:
    def __init__(self, V, D=32, L=2, T=16, M=64):
        self.V, self.D, self.L, self.T, self.M = V, D, L, T, M

def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xn = xc * inv
    return xn * g + b, (xn, inv, g)

def layernorm_bwd(dy, cache, x):
    xn, inv, g = cache
    D = x.shape[-1]
    dg = (dy * xn).sum(0)
    db = dy.sum(0)
    dxn = dy * g
    dx = inv * (dxn - dxn.mean(-1, keepdims=True) - xn * (dxn * xn).mean(-1, keepdims=True))
    return dx, dg, db

def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

class GPT:
    def __init__(self, cfg, seed=0):
        r = np.random.RandomState(seed)
        D, V, M, T, L = cfg.D, cfg.V, cfg.M, cfg.T, cfg.L
        s = 0.02
        self.cfg = cfg
        self.p = {
            'tok': r.randn(V, D) * s,
            'pos': r.randn(T, D) * s,
        }
        for l in range(L):
            self.p[f'ln1g{l}'] = np.ones(D);  self.p[f'ln1b{l}'] = np.zeros(D)
            self.p[f'Wq{l}'] = r.randn(D, D)*s; self.p[f'bq{l}'] = np.zeros(D)
            self.p[f'Wk{l}'] = r.randn(D, D)*s; self.p[f'bk{l}'] = np.zeros(D)
            self.p[f'Wv{l}'] = r.randn(D, D)*s; self.p[f'bv{l}'] = np.zeros(D)
            self.p[f'Wo{l}'] = r.randn(D, D)*s; self.p[f'bo{l}'] = np.zeros(D)
            self.p[f'ln2g{l}'] = np.ones(D);  self.p[f'ln2b{l}'] = np.zeros(D)
            self.p[f'W1{l}'] = r.randn(D, M)*s; self.p[f'b1{l}'] = np.zeros(M)
            self.p[f'W2{l}'] = r.randn(M, D)*s; self.p[f'b2{l}'] = np.zeros(D)
        self.p['lnfg'] = np.ones(D); self.p['lnfb'] = np.zeros(D)
        self.p['Wh'] = r.randn(D, V)*s; self.p['bh'] = np.zeros(V)

    def forward(self, idx, cache=None):
        """idx: (T,) int.  Returns logits (T,V).  Fills cache if given."""
        cfg = self.cfg; D = cfg.D; p = self.p
        T = len(idx)
        x = p['tok'][idx] + p['pos'][:T]
        mask = np.triu(np.ones((T, T)), 1).astype(bool)
        for l in range(cfg.L):
            h, cln1 = layernorm(x, p[f'ln1g{l}'], p[f'ln1b{l}'])
            q = h @ p[f'Wq{l}'] + p[f'bq{l}']
            k = h @ p[f'Wk{l}'] + p[f'bk{l}']
            v = h @ p[f'Wv{l}'] + p[f'bv{l}']
            sc = (q @ k.T) / np.sqrt(D)
            sc = np.where(mask, -1e9, sc)
            a = softmax(sc)
            o = a @ v
            o = o @ p[f'Wo{l}'] + p[f'bo{l}']
            x = x + o
            h2, cln2 = layernorm(x, p[f'ln2g{l}'], p[f'ln2b{l}'])
            z1 = h2 @ p[f'W1{l}'] + p[f'b1{l}']
            a1 = np.maximum(z1, 0)
            m = a1 @ p[f'W2{l}'] + p[f'b2{l}']
            x = x + m
            if cache is not None:
                cache[l] = dict(h=h, cln1=cln1, q=q, k=k, v=v, a=a, o_pre=a@v,
                                x_after_attn=x - m, h2=h2, cln2=cln2, z1=z1, a1=a1)
        xf, clnf = layernorm(x, p['lnfg'], p['lnfb'])
        logits = xf @ p['Wh'] + p['bh']
        if cache is not None:
            cache['xf'] = xf; cache['clnf'] = clnf; cache['x_final'] = x
            cache['idx'] = idx
        return logits

    def loss_and_grad(self, idx, tgt):
        cfg = self.cfg; D = cfg.D; p = self.p; T = len(idx)
        cache = {}
        logits = self.forward(idx, cache)
        # cross entropy
        pr = softmax(logits)
        loss = -np.log(pr[np.arange(T), tgt] + 1e-12).mean()
        g = {key: np.zeros_like(val) for key, val in p.items()}
        dlogits = pr.copy(); dlogits[np.arange(T), tgt] -= 1; dlogits /= T
        # head
        g['Wh'] += cache['xf'].T @ dlogits
        g['bh'] += dlogits.sum(0)
        dxf = dlogits @ p['Wh'].T
        dx, dg, db = layernorm_bwd(dxf, cache['clnf'], cache['x_final'])
        g['lnfg'] += dg; g['lnfb'] += db
        mask = np.triu(np.ones((T, T)), 1).astype(bool)
        for l in reversed(range(cfg.L)):
            c = cache[l]
            # mlp branch: x = x + m ;  m = relu(h2@W1+b1)@W2+b2
            dm = dx.copy()
            g[f'W2{l}'] += c['a1'].T @ dm
            g[f'b2{l}'] += dm.sum(0)
            da1 = dm @ p[f'W2{l}'].T
            dz1 = da1 * (c['z1'] > 0)
            g[f'W1{l}'] += c['h2'].T @ dz1
            g[f'b1{l}'] += dz1.sum(0)
            dh2 = dz1 @ p[f'W1{l}'].T
            dx_ln2, dg2, db2 = layernorm_bwd(dh2, c['cln2'], c['x_after_attn'])
            g[f'ln2g{l}'] += dg2; g[f'ln2b{l}'] += db2
            dx = dx + dx_ln2          # residual around mlp
            # attn branch: x = x + o ; o=(a@v)@Wo+bo
            do = dx.copy()
            g[f'Wo{l}'] += c['o_pre'].T @ do
            g[f'bo{l}'] += do.sum(0)
            dov = do @ p[f'Wo{l}'].T      # d(a@v)
            da = dov @ c['v'].T
            dv = c['a'].T @ dov
            # softmax bwd
            dsc = c['a'] * (da - (da * c['a']).sum(-1, keepdims=True))
            dsc = np.where(mask, 0.0, dsc) / np.sqrt(D)
            dq = dsc @ c['k']
            dk = dsc.T @ c['q']
            g[f'Wq{l}'] += c['h'].T @ dq; g[f'bq{l}'] += dq.sum(0)
            g[f'Wk{l}'] += c['h'].T @ dk; g[f'bk{l}'] += dk.sum(0)
            g[f'Wv{l}'] += c['h'].T @ dv; g[f'bv{l}'] += dv.sum(0)
            dh = dq @ p[f'Wq{l}'].T + dk @ p[f'Wk{l}'].T + dv @ p[f'Wv{l}'].T
            dx_ln1, dg1, db1 = layernorm_bwd(dh, c['cln1'], None_to_h(c))
            g[f'ln1g{l}'] += dg1; g[f'ln1b{l}'] += db1
            dx = dx + dx_ln1          # residual around attn
        # embeddings
        idxa = cache['idx']
        for t in range(T):
            g['tok'][idxa[t]] += dx[t]
        g['pos'][:T] += dx
        return loss, g

def None_to_h(c):
    # LN1 input was the block input x; reconstruct it from cache: h was LN(x_in).
    # We need x_in for layernorm_bwd's shape only (it uses cache, not x values
    # except shape), so any array with right shape works.
    return c['h']
