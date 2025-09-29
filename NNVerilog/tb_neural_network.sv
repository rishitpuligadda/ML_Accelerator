`timescale 1ns/1ps

module tb_neural_network_2layer;

    localparam IN_SIZE  = 4;
    localparam HIDDEN1  = 3;
    localparam OUT_SIZE = 2;
    localparam WIDTH    = 16;
    localparam FRAC     = 8;

    logic signed [WIDTH-1:0] in_vec  [1][IN_SIZE];
    logic signed [WIDTH-1:0] W1      [HIDDEN1][IN_SIZE];
    logic signed [WIDTH-1:0] W2      [OUT_SIZE][HIDDEN1];
    logic signed [WIDTH-1:0] out_vec [1][OUT_SIZE];

    // Helpers to convert between real and fixed-point
    function automatic logic signed [WIDTH-1:0] to_fixed(input real val);
        to_fixed = $rtoi(val * (1 << FRAC));
    endfunction

    function automatic real to_real(input logic signed [WIDTH-1:0] val);
        to_real = val / real'(1 << FRAC);
    endfunction

    // Instantiate neural network
    neural_network_2layer #(
        .IN_SIZE(IN_SIZE),
        .HIDDEN1(HIDDEN1),
        .OUT_SIZE(OUT_SIZE),
        .WIDTH(WIDTH),
        .FRAC(FRAC)
    ) dut (
        .in_vec(in_vec),
        .W1(W1),
        .W2(W2),
        .out_vec(out_vec)
    );

    initial begin
        // ---- Example weights ----
        W1[0] = '{to_fixed(-0.05002382), to_fixed(-0.18386332), to_fixed(0.04406993), to_fixed(0.25553058)};
        W1[1] = '{to_fixed(0.08154866),  to_fixed(-0.0703455),  to_fixed(0.08233125), to_fixed(0.01279148)};
        W1[2] = '{to_fixed(0.06219365),  to_fixed(0.2212559),   to_fixed(-0.01675784), to_fixed(0.09114323)};

        W2[0] = '{to_fixed(-0.02116636), to_fixed(-0.02683857), to_fixed(-0.06972045)};
        W2[1] = '{to_fixed(0.07114158),  to_fixed(0.14845954),  to_fixed(0.07781072)};
        // ---- Input sample 0 ----
        in_vec[0] = '{to_fixed(1), to_fixed(2), to_fixed(3), to_fixed(4)};
        #10;
        $display("Sample 0 Output:");
        for(int i=0; i<OUT_SIZE; i++)
            $display("  out[%0d] = %0d / %f", i, out_vec[0][i], to_real(out_vec[0][i]));

        // ---- Input sample 1 ----
        in_vec[0] = '{to_fixed(2), to_fixed(-1), to_fixed(0), to_fixed(3)};
        #10;
        $display("Sample 1 Output:");
        for(int i=0; i<OUT_SIZE; i++)
            $display("  out[%0d] = %0d / %f", i, out_vec[0][i], to_real(out_vec[0][i]));

        #10 $finish;
    end

endmodule

