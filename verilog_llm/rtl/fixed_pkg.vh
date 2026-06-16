// Fixed-point Q(.FRAC) primitives.  Included *inside* module `gpt` so the
// functions can see localparams W, FRAC, ONE, MAXV, MINV.  Every function is a
// bt-for-bt port of the corresponding routine in ref/fixed.py.

// saturate a wide signed value to the 32-bt word range
function signed [W-1:0] sat32(input signed [63:0] x);
  begin
    if (x > MAXV)      sat32 = MAXV[W-1:0];
    else if (x < MINV) sat32 = MINV[W-1:0];
    else               sat32 = x[W-1:0];
  end
endfunction

// Q*Q -> Q, arithmetic right shift (floor), saturated
function signed [W-1:0] fmul(input signed [W-1:0] a, input signed [W-1:0] b);
  reg signed [63:0] p;
  begin
    p = $signed(a) * $signed(b);   // 64-bt product
    fmul = sat32(p >>> FRAC);
  end
endfunction

// saturated add / sub
function signed [W-1:0] fadd(input signed [W-1:0] a, input signed [W-1:0] b);
  begin fadd = sat32($signed(a) + $signed(b)); end
endfunction
function signed [W-1:0] fsub(input signed [W-1:0] a, input signed [W-1:0] b);
  begin fsub = sat32($signed(a) - $signed(b)); end
endfunction

// Q/Q -> Q, truncating toward zero (matches Verilog signed '/')
function signed [W-1:0] fdiv(input signed [W-1:0] a, input signed [W-1:0] b);
  reg signed [63:0] num;
  begin
    num  = $signed(a) <<< FRAC;
    fdiv = sat32(num / $signed({{32{b[W-1]}}, b}));
  end
endfunction

// integer floor(sqrt(n)) for n >= 0  (classic bt-by-bt algorithm)
function [W-1:0] isqrt(input [63:0] n);
  reg [63:0] num, res, bt;
  begin
    num = n; res = 0;
    bt = 64'h4000000000000000;       // 1 << 62  (highest even power of 4)
    while (bt > num) bt = bt >> 2;
    while (bt != 0) begin
      if (num >= res + bt) begin
        num = num - (res + bt);
        res = (res >> 1) + bt;
      end else
        res = res >> 1;
      bt = bt >> 2;
    end
    isqrt = res[W-1:0];
  end
endfunction

// sqrt of a Q number, returned as Q
function signed [W-1:0] fsqrt(input signed [W-1:0] x);
  begin
    if (x <= 0) fsqrt = 0;
    else        fsqrt = isqrt({32'd0, x} <<< FRAC);
  end
endfunction

// 1/sqrt(x) in Q
function signed [W-1:0] frsqrt(input signed [W-1:0] x);
  reg signed [W-1:0] s;
  begin
    s = fsqrt(x);
    if (s == 0) frsqrt = MAXV[W-1:0];
    else        frsqrt = fdiv(ONE[W-1:0], s);
  end
endfunction

// exp(z) for z <= 0, returned as Q in [0, ONE].  exp(z)=2^(z*log2e), split into
// integer + fractional parts, 2^frac via cubic polynomial (Taylor of 2^f).
function signed [W-1:0] fexp_neg(input signed [W-1:0] zin);
  reg signed [W-1:0] z, t, f, p;
  integer k, shift;
  begin
    z = (zin > 0) ? 0 : zin;
    t = fmul(z, `LOG2E);             // <= 0
    k = t >>> FRAC;                  // floor (negative)
    f = t - (k <<< FRAC);            // fractional part in [0, ONE)
    p = fadd(fmul(`EXP_P3, f), `EXP_P2);
    p = fadd(fmul(p,        f), `EXP_P1);
    p = fadd(fmul(p,        f), `EXP_P0);
    shift = -k;
    if (shift >= W) fexp_neg = 0;
    else            fexp_neg = p >>> shift;
  end
endfunction
