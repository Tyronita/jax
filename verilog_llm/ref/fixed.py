"""Bit-accurate fixed-point primitives shared with the Verilog RTL.

Everything here operates on Python ints that represent signed Q(INT.FRAC)
fixed-point numbers (value_real = value_int / 2**FRAC).  The exact same integer
operations are implemented in rtl/fixed_pkg.vh so the hardware reproduces these
results bit-for-bit.
"""

W    = 32          # word width (bits)
FRAC = 16          # fractional bits  -> Q16.16
ONE  = 1 << FRAC   # 1.0 in fixed point
MINV = -(1 << (W - 1))
MAXV =  (1 << (W - 1)) - 1

# constants used by exp()
LOG2E = round(1.4426950408889634 * ONE)   # log2(e)

def sat(x):
    """Saturate to the signed word range (matches Verilog wire width)."""
    if x > MAXV: return MAXV
    if x < MINV: return MINV
    return x

def to_fixed(x):
    return sat(int(round(x * ONE)))

def to_float(x):
    return x / ONE

def fmul(a, b):
    """Q*Q -> Q with arithmetic right shift (truncation toward -inf)."""
    return sat((a * b) >> FRAC)

def fdiv(a, b):
    """Q/Q -> Q.  Truncating signed division (matches Verilog '/')."""
    num = a << FRAC
    # Verilog '/' truncates toward zero; mirror that here.
    q = abs(num) // abs(b)
    if (num < 0) ^ (b < 0):
        q = -q
    return sat(q)

def isqrt(n):
    """Integer floor(sqrt(n)) via the classic bit-by-bit algorithm
    (identical loop is used in fixed_pkg.vh)."""
    if n <= 0:
        return 0
    res = 0
    bit = 1 << ((n.bit_length() - 1) & ~1)   # highest even power of 4 <= n
    while bit > n:
        bit >>= 2
    while bit != 0:
        if n >= res + bit:
            n -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2
    return res

def fsqrt(x):
    """sqrt of a Q number, returned as Q:  sqrt(x)*2^FRAC = isqrt(x<<FRAC)."""
    if x <= 0:
        return 0
    return isqrt(x << FRAC)

def frsqrt(x):
    """1/sqrt(x) in Q, via fsqrt then reciprocal."""
    s = fsqrt(x)
    if s == 0:
        return MAXV
    return fdiv(ONE, s)

# --- exp() for z <= 0 (the only range softmax needs after subtracting max) ---
# exp(z) = 2^(z*log2e).  Split into integer (k) and fractional (f) parts and use
# a cubic polynomial for 2^f, f in [0,1).  Poly coeffs are the Taylor series of
# 2^f = e^{f ln2}.  All integer; reproduced exactly in hardware.
P0 = ONE
P1 = round(0.6931471805599453 * ONE)
P2 = round(0.2402265069591007 * ONE)
P3 = round(0.0555041086648216 * ONE)

def fexp_neg(z):
    """exp(z) in Q for z <= 0.  Returns 0..ONE."""
    if z > 0:
        z = 0
    t = fmul(z, LOG2E)          # t = z*log2e  (<= 0), Q
    k = t >> FRAC               # floor(t), negative integer
    f = t - (k << FRAC)         # fractional part in [0,ONE)
    # 2^f via Horner: ((P3*f + P2)*f + P1)*f + P0
    p = fmul(P3, f) + P2
    p = fmul(p, f) + P1
    p = fmul(p, f) + P0         # ~ 2^f in [ONE, 2*ONE)
    shift = -k                  # >= 0
    if shift >= W:
        return 0
    return p >> shift
