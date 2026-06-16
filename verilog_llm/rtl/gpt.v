// =====================================================================
//  gpt.v  --  a tiny GPT-style language model in synthesizable Verilog
//
//  Decoder-only transformer, single attention head, pre-norm, ReLU MLP.
//  Q16.16 fixed-point arithmetic.  Weights are trained in PyTorch-free
//  numpy (ref/) and loaded from .hex via $readmemh.  The module loads a
//  prompt and autoregressively generates tokens (greedy/argmax) until the
//  sequence buffer is full, asserting `done`.
//
//  Verified bit-exact against ref/fixed_infer.py (the golden model).
// =====================================================================
`include "config.vh"

module gpt (
    input  wire clk,
    input  wire rst,
    input  wire start,
    output reg  done
);
    // ---- dimensions / fixed-point params ----
    localparam V    = `VOCAB;
    localparam D    = `DMODEL;
    localparam L    = `NLAYER;
    localparam T    = `SEQLEN;
    localparam M    = `DFF;
    localparam W    = `WBITS;
    localparam FRAC = `FRAC;
    localparam P    = `PROMPT_LEN;
    localparam signed [63:0] ONE  =  (64'd1 <<< FRAC);
    localparam signed [63:0] MAXV =  (64'd1 <<< (W-1)) - 1;
    localparam signed [63:0] MINV = -(64'd1 <<< (W-1));

    `include "fixed_pkg.vh"

    // ---- weight memories ----
    reg signed [W-1:0] tok_mem  [0:V*D-1];
    reg signed [W-1:0] pos_mem  [0:T*D-1];
    reg signed [W-1:0] Wq_mem   [0:L*D*D-1];
    reg signed [W-1:0] Wk_mem   [0:L*D*D-1];
    reg signed [W-1:0] Wv_mem   [0:L*D*D-1];
    reg signed [W-1:0] Wo_mem   [0:L*D*D-1];
    reg signed [W-1:0] W1_mem   [0:L*M*D-1];
    reg signed [W-1:0] W2_mem   [0:L*D*M-1];
    reg signed [W-1:0] bq_mem   [0:L*D-1];
    reg signed [W-1:0] bk_mem   [0:L*D-1];
    reg signed [W-1:0] bv_mem   [0:L*D-1];
    reg signed [W-1:0] bo_mem   [0:L*D-1];
    reg signed [W-1:0] b1_mem   [0:L*M-1];
    reg signed [W-1:0] b2_mem   [0:L*D-1];
    reg signed [W-1:0] ln1g_mem [0:L*D-1];
    reg signed [W-1:0] ln1b_mem [0:L*D-1];
    reg signed [W-1:0] ln2g_mem [0:L*D-1];
    reg signed [W-1:0] ln2b_mem [0:L*D-1];
    reg signed [W-1:0] lnfg_mem [0:D-1];
    reg signed [W-1:0] lnfb_mem [0:D-1];
    reg signed [W-1:0] Wh_mem   [0:V*D-1];
    reg signed [W-1:0] bh_mem   [0:V-1];

    // ---- activation scratch ----
    reg signed [W-1:0] xb   [0:T*D-1];   // residual stream
    reg signed [W-1:0] hb   [0:T*D-1];   // ln output (ln1 / final)
    reg signed [W-1:0] h2b  [0:T*D-1];   // ln2 output
    reg signed [W-1:0] qb   [0:T*D-1];
    reg signed [W-1:0] kb   [0:T*D-1];
    reg signed [W-1:0] vb   [0:T*D-1];
    reg signed [W-1:0] ob   [0:T*D-1];
    reg signed [W-1:0] opb  [0:T*D-1];
    reg signed [W-1:0] z1b  [0:M-1];
    reg signed [W-1:0] a1b  [0:M-1];
    reg signed [W-1:0] mvb  [0:D-1];
    reg signed [W-1:0] scr  [0:T-1];
    reg signed [W-1:0] er   [0:T-1];
    reg signed [W-1:0] ar   [0:T-1];
    reg signed [W-1:0] lntmp[0:D-1];
    reg signed [W-1:0] logits_mem [0:T*V-1];

    // ---- generation state ----
    reg signed [W-1:0] seq [0:T-1];      // token ids (plain integers)
    integer cur;
    reg [1:0] st;
    localparam S_IDLE=0, S_GEN=1, S_DONE=2;

    reg signed [W-1:0] prompt_mem [0:P-1];
    integer ii;
    initial begin
        `include "load_weights.vh"
        $readmemh("weights/prompt.hex", prompt_mem);
        for (ii=0; ii<P; ii=ii+1) seq[ii] = prompt_mem[ii];
        done = 0; st = S_IDLE;
    end

    // ---- layernorm: writes normalized (xc * 1/sqrt(vr+eps)) into lntmp ----
    task ln_norm(input integer src);
        integer i; reg signed [63:0] s, sv; reg signed [W-1:0] mean, vr, inv;
        begin
            s = 0;
            for (i=0;i<D;i=i+1) s = s + $signed(xb[src+i]);
            mean = sat32(s / D);
            sv = 0;
            for (i=0;i<D;i=i+1) begin
                lntmp[i] = fsub(xb[src+i], mean);          // xc
                sv = sv + $signed(fmul(lntmp[i], lntmp[i]));
            end
            vr = sat32(sv / D);
            inv = frsqrt(fadd(vr, `EPS_Q));
            for (i=0;i<D;i=i+1) lntmp[i] = fmul(lntmp[i], inv);
        end
    endtask

    // ---- one full forward pass over seq -> logits_mem ----
    task forward;
        integer l, t, j, d, o, kk;
        reg signed [63:0] acc, tot;
        reg signed [W-1:0] mx, v;
        begin
            // token + positional embeddings
            for (t=0;t<T;t=t+1)
                for (d=0;d<D;d=d+1)
                    xb[t*D+d] = fadd(tok_mem[seq[t]*D+d], pos_mem[t*D+d]);

            for (l=0;l<L;l=l+1) begin
                // ---- attention block ----
                for (t=0;t<T;t=t+1) begin              // LN1
                    ln_norm(t*D);
                    for (d=0;d<D;d=d+1)
                        hb[t*D+d] = fadd(fmul(lntmp[d], ln1g_mem[l*D+d]), ln1b_mem[l*D+d]);
                end
                for (t=0;t<T;t=t+1)                     // Q,K,V projections
                    for (o=0;o<D;o=o+1) begin
                        acc=0;
                        for (kk=0;kk<D;kk=kk+1) acc=acc+$signed(Wq_mem[l*D*D+o*D+kk])*$signed(hb[t*D+kk]);
                        qb[t*D+o]=fadd(sat32(acc>>>FRAC), bq_mem[l*D+o]);
                        acc=0;
                        for (kk=0;kk<D;kk=kk+1) acc=acc+$signed(Wk_mem[l*D*D+o*D+kk])*$signed(hb[t*D+kk]);
                        kb[t*D+o]=fadd(sat32(acc>>>FRAC), bk_mem[l*D+o]);
                        acc=0;
                        for (kk=0;kk<D;kk=kk+1) acc=acc+$signed(Wv_mem[l*D*D+o*D+kk])*$signed(hb[t*D+kk]);
                        vb[t*D+o]=fadd(sat32(acc>>>FRAC), bv_mem[l*D+o]);
                    end
                for (t=0;t<T;t=t+1) begin               // causal self-attention
                    for (j=0;j<=t;j=j+1) begin          // scores q.k
                        acc=0;
                        for (d=0;d<D;d=d+1) acc=acc+$signed(qb[t*D+d])*$signed(kb[j*D+d]);
                        scr[j]=fmul(sat32(acc>>>FRAC), `INV_SQRT_D);
                    end
                    mx=scr[0];
                    for (j=1;j<=t;j=j+1) if (scr[j]>mx) mx=scr[j];
                    tot=0;
                    for (j=0;j<=t;j=j+1) begin er[j]=fexp_neg(fsub(scr[j],mx)); tot=tot+$signed(er[j]); end
                    if (tot==0) tot=1;
                    for (j=0;j<=t;j=j+1) ar[j]=fdiv(er[j], sat32(tot));
                    for (d=0;d<D;d=d+1) begin            // weighted sum of V
                        acc=0;
                        for (j=0;j<=t;j=j+1) acc=acc+$signed(ar[j])*$signed(vb[j*D+d]);
                        ob[t*D+d]=sat32(acc>>>FRAC);
                    end
                end
                for (t=0;t<T;t=t+1)                      // output projection
                    for (o=0;o<D;o=o+1) begin
                        acc=0;
                        for (kk=0;kk<D;kk=kk+1) acc=acc+$signed(Wo_mem[l*D*D+o*D+kk])*$signed(ob[t*D+kk]);
                        opb[t*D+o]=fadd(sat32(acc>>>FRAC), bo_mem[l*D+o]);
                    end
                for (t=0;t<T;t=t+1)                      // residual
                    for (d=0;d<D;d=d+1) xb[t*D+d]=fadd(xb[t*D+d], opb[t*D+d]);

                // ---- MLP block ----
                for (t=0;t<T;t=t+1) begin               // LN2
                    ln_norm(t*D);
                    for (d=0;d<D;d=d+1)
                        h2b[t*D+d]=fadd(fmul(lntmp[d], ln2g_mem[l*D+d]), ln2b_mem[l*D+d]);
                end
                for (t=0;t<T;t=t+1) begin
                    for (o=0;o<M;o=o+1) begin            // W1 + ReLU
                        acc=0;
                        for (kk=0;kk<D;kk=kk+1) acc=acc+$signed(W1_mem[l*M*D+o*D+kk])*$signed(h2b[t*D+kk]);
                        v=fadd(sat32(acc>>>FRAC), b1_mem[l*M+o]);
                        a1b[o]=(v>0)?v:0;
                    end
                    for (o=0;o<D;o=o+1) begin            // W2
                        acc=0;
                        for (kk=0;kk<M;kk=kk+1) acc=acc+$signed(W2_mem[l*D*M+o*M+kk])*$signed(a1b[kk]);
                        mvb[o]=fadd(sat32(acc>>>FRAC), b2_mem[l*D+o]);
                    end
                    for (d=0;d<D;d=d+1) xb[t*D+d]=fadd(xb[t*D+d], mvb[d]);
                end
            end

            // ---- final LN + output head ----
            for (t=0;t<T;t=t+1) begin
                ln_norm(t*D);
                for (d=0;d<D;d=d+1)
                    hb[t*D+d]=fadd(fmul(lntmp[d], lnfg_mem[d]), lnfb_mem[d]);
            end
            for (t=0;t<T;t=t+1)
                for (o=0;o<V;o=o+1) begin
                    acc=0;
                    for (kk=0;kk<D;kk=kk+1) acc=acc+$signed(Wh_mem[o*D+kk])*$signed(hb[t*D+kk]);
                    logits_mem[t*V+o]=fadd(sat32(acc>>>FRAC), bh_mem[o]);
                end
        end
    endtask

    // ---- generation FSM: one token per clock ----
    integer o2; reg signed [W-1:0] best; integer bi;
    always @(posedge clk) begin
        if (rst) begin
            st <= S_IDLE; done <= 0;
        end else case (st)
            S_IDLE: if (start) begin
                        for (ii=P; ii<T; ii=ii+1) seq[ii] = 0;   // clear tail
                        cur = P; done <= 0; st <= S_GEN;
                    end
            S_GEN: begin
                        forward();
                        best = logits_mem[(cur-1)*V+0]; bi = 0;
                        for (o2=1;o2<V;o2=o2+1)
                            if (logits_mem[(cur-1)*V+o2] > best) begin
                                best = logits_mem[(cur-1)*V+o2]; bi = o2;
                            end
                        seq[cur] = bi;
                        if (cur+1 >= T) st <= S_DONE;
                        else            cur = cur + 1;
                    end
            S_DONE: done <= 1;
        endcase
    end
endmodule
