// Testbench: run the hardware LLM, print the text it generates, and check it
// matches the golden token stream from ref/fixed_infer.py.
`timescale 1ns/1ps
`include "config.vh"

module tb_gpt;
    localparam T = `SEQLEN;
    localparam V = `VOCAB;

    `include "id2asc.vh"

    reg clk = 0, rst = 1, start = 0;
    wire done;
    always #5 clk = ~clk;

    gpt dut (.clk(clk), .rst(rst), .start(start), .done(done));

    reg signed [`WBITS-1:0] golden [0:T-1];
    integer i, errors;

    initial begin
        $readmemh("weights/golden.hex", golden);
        @(negedge clk); @(negedge clk); rst = 0;
        @(negedge clk); start = 1; @(negedge clk); start = 0;
        wait (done);
        @(negedge clk);

        $write("\n  generated: \"");
        for (i=0;i<T;i=i+1) $write("%c", id2asc(dut.seq[i]));
        $write("\"\n");

        errors = 0;
        for (i=0;i<T;i=i+1)
            if (dut.seq[i] !== golden[i]) begin
                errors = errors + 1;
                $display("  MISMATCH pos %0d: hw=%0d golden=%0d", i, dut.seq[i], golden[i]);
            end
        if (errors==0)
            $display("  PASS: hardware output is bit-exact with the golden model (%0d tokens).", T);
        else
            $display("  FAIL: %0d mismatches.", errors);
        $finish;
    end

    initial begin  // safety timeout
        #5000000 $display("TIMEOUT"); $finish;
    end
endmodule
