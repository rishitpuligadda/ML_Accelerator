`timescale 1ns/1ps

module tb_dense_layer;
    parameter B = 2;
    parameter M = 3;
    parameter N = 4;
    parameter  WIDTH = 16;

    logic signed [WIDTH-1:0] weights [M][N];
    logic signed [WIDTH-1:0] inputs [B][N];
    logic signed [2*WIDTH-1:0] result [B][M];

    dense_layer #(.B(B), .M(M), .N(N), .WIDTH(WIDTH)) dut
        (.weights(weights), .inputs(inputs), .result(result));

    initial begin
        weights[0] = '{1, 2, 3, 4};
        weights[1] = '{-1, 0, 2, 1};
        weights[2] = '{3, -2, 1, 0};

        inputs[0] = '{1, 2, 3, 4};
        inputs[1] = '{2, -1, 0, 3};

        #1;

        for(int b = 0; b < B; b++) begin
            $display("Batch[%0d] Inputs = %p", b, inputs[b]);
            for(int i = 0; i < M; i++) begin
                $display("  Neuron[%0d]: Output = %0d", i, result[b][i]);
            end
        end
        #5 $finish;
    end

endmodule
