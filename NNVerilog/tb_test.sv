`timescale 1ns/1ps

module tb_exp_approx;

    // Parameters
    localparam WIDTH = 16;
    localparam FRAC  = 8;

    // Signals
    logic signed [WIDTH-1:0] x;
    logic signed [WIDTH-1:0] y;

    // Instantiate DUT
    exp_approx #(.WIDTH(WIDTH), .FRAC(FRAC)) dut (
        .x(x),
        .y(y)
    );

    // Helper functions
    function automatic logic signed [WIDTH-1:0] to_fixed(input real val);
        to_fixed = $rtoi(val * (1 << FRAC));
    endfunction

    function automatic real to_real(input logic signed [WIDTH-1:0] val);
        to_real = val / real'(1 << FRAC);
    endfunction

    // Test
    initial begin
        automatic real test_val = -1.2;

        // Convert to fixed-point
        x = to_fixed(test_val);
        #10; // wait for combinational logic to settle

        $display("Input: %f (fixed: %0d)", test_val, x);
        $display("Output (fixed): %0d, Output (real): %f", y, to_real(y));

        #10 $finish;
    end

endmodule

