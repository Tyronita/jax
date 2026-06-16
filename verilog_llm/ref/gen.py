"""Generate text from the exported (quantized) weights using the bit-exact
fixed-point model -- i.e. exactly what the Verilog computes.

Usage:  python3 gen.py "the quick "
"""
import os, sys, json
import fixed as F
from fixed_infer import FixedGPT

HERE = os.path.dirname(__file__)
WDIR = os.path.abspath(os.path.join(HERE, '..', 'weights'))

meta = json.load(open(os.path.join(WDIR, 'meta.json')))
cfg, consts = meta['cfg'], meta['consts']
stoi = meta['stoi']; itos = {int(k):v for k,v in meta['itos'].items()}

def load(name):
    out = []
    for line in open(os.path.join(WDIR, name+'.hex')):
        x = int(line.strip(), 16)
        if x >= (1 << 31): x -= (1 << 32)   # two's complement -> signed
        out.append(x)
    return out

q = {n: load(n) for n in meta['names']}
fg = FixedGPT(q, cfg, consts)

prompt_s = sys.argv[1] if len(sys.argv) > 1 else "the "
prompt = [stoi[c] for c in prompt_s]
prompt = prompt[:cfg['T']]
gen = fg.generate(prompt, cfg['T'])
print(repr(''.join(itos[t] for t in gen)))
