`timescale 1ns/1ps

module tb_dense_layer;
    parameter B = 2;
    parameter M = 3;
    parameter N = 4;
    parameter WIDTH = 16;
    parameter FRAC  = 8;   // Q8.8 fixed-point

    logic signed [WIDTH-1:0] weights [M][N];
    logic signed [WIDTH-1:0] inputs  [B][N];
    logic signed [WIDTH-1:0] result  [B][M];

    // Instantiate DUT
    dense_layer #(.B(B), .M(M), .N(N), .WIDTH(WIDTH), .FRAC(FRAC)) dut (
        .weights(weights), 
        .inputs(inputs), 
        .result(result)
    );

    // Function to convert real -> fixed-point
    function automatic logic signed [WIDTH-1:0] to_fixed(input real val);
        to_fixed = $rtoi(val * (1 << FRAC));
    endfunction

    // Function to convert fixed-point -> real
    function automatic real to_real(input logic signed [WIDTH-1:0] val);
        to_real = val / real'(1 << FRAC);
    endfunction

    initial begin
        // Assign weights (real values converted to fixed-point)
        weights[0] = '{ to_fixed(0.03779225),  to_fixed(-0.0181641),  to_fixed(-0.12103897),  to_fixed(-0.04087051) };
        weights[1] = '{ to_fixed(0.06265889), to_fixed(0.0380107),  to_fixed(-0.01975206),  to_fixed(0.16375556) };
        weights[2] = '{ to_fixed(0.0398717),  to_fixed(0.01922126), to_fixed(0.09439117),  to_fixed(-0.13749411) };

        // Assign inputs (real values converted to fixed-point)
        inputs[0] = '{ to_fixed(1.0), to_fixed(2.0), to_fixed(3.0), to_fixed(4.0) };
        inputs[1] = '{ to_fixed(2.0), to_fixed(-1.0), to_fixed(0.0), to_fixed(3.0) };

        #1;

        // Display results
        for(int b = 0; b < B; b++) begin
            $display("Batch[%0d] Inputs (fixed) = %p", b, inputs[b]);
            for(int i = 0; i < M; i++) begin
                $display("  Neuron[%0d]: Output = %0d (fixed), %f (real)", 
                          i, result[b][i], to_real(result[b][i]));
            end
        end

        #5 $finish;
    end
endmodule
