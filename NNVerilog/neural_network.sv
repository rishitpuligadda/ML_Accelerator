`timescale 1ns/1ps

module neural_network_2layer_softmax #(
    parameter IN_SIZE    = 2,
    parameter HIDDEN1    = 64,
    parameter OUT_SIZE   = 3,
    parameter WIDTH      = 16,
    parameter FRAC       = 8,
    parameter BATCH      = 15
)(
    input  logic signed [WIDTH-1:0] in_vec     [BATCH][IN_SIZE],
    input  logic signed [WIDTH-1:0] W1         [HIDDEN1][IN_SIZE],
    input  logic signed [WIDTH-1:0] B1         [HIDDEN1],
    input  logic signed [WIDTH-1:0] W2         [OUT_SIZE][HIDDEN1],
    input  logic signed [WIDTH-1:0] B2         [OUT_SIZE],
    output logic signed [WIDTH-1:0] softmax_out[BATCH][OUT_SIZE],

    // Debug port: only expose Layer 2 outputs
    output logic signed [WIDTH-1:0] debug_dense2[BATCH][OUT_SIZE]
);

    logic signed [WIDTH-1:0] dense1_out [BATCH][HIDDEN1];
    logic signed [WIDTH-1:0] relu1_out  [BATCH][HIDDEN1];
    logic signed [WIDTH-1:0] dense2_out [BATCH][OUT_SIZE];

    // ---- Layer 1: Dense + Bias ----
    dense_layer #(
        .B(BATCH), .M(HIDDEN1), .N(IN_SIZE), .WIDTH(WIDTH), .FRAC(FRAC)
    ) layer1 (
        .weights(W1),
        .bias(B1),
        .inputs(in_vec),
        .result(dense1_out)
    );

    // ---- Layer 1: ReLU ----
    genvar b, i;
    generate
        for (b = 0; b < BATCH; b++) begin : batch_loop1
            for (i = 0; i < HIDDEN1; i++) begin : neuron_loop1
                relu #(.WIDTH(WIDTH), .FRAC(FRAC)) relu_inst (
                    .in(dense1_out[b][i]),
                    .out(relu1_out[b][i])
                );
            end
        end
    endgenerate

    // ---- Layer 2: Dense + Bias (logits) ----
    dense_layer #(
        .B(BATCH), .M(OUT_SIZE), .N(HIDDEN1), .WIDTH(WIDTH), .FRAC(FRAC)
    ) layer2 (
        .weights(W2),
        .bias(B2),
        .inputs(relu1_out),
        .result(dense2_out)
    );

    // Connect Layer 2 outputs to debug port
    assign debug_dense2 = dense2_out;

    // ---- Softmax ----
    generate
        for (b = 0; b < BATCH; b++) begin : batch_softmax
            softmax_fixedpt #(
                .WIDTH(WIDTH),
                .FRAC(FRAC),
                .SIZE(OUT_SIZE)
            ) sm (
                .in(dense2_out[b]),
                .out(softmax_out[b])
            );
        end
    endgenerate

endmodule


module neural_network_3layer_softmax #(
    parameter IN_SIZE    = 784,
    parameter HIDDEN1    = 128,
    parameter HIDDEN2    = 128,
    parameter OUT_SIZE   = 10,
    parameter WIDTH      = 16,
    parameter FRAC       = 8,
    parameter BATCH      = 15
)(
    input  logic signed [WIDTH-1:0] in_vec      [BATCH][IN_SIZE],
    input  logic signed [WIDTH-1:0] W1          [HIDDEN1][IN_SIZE],
    input  logic signed [WIDTH-1:0] B1          [HIDDEN1],
    input  logic signed [WIDTH-1:0] W2          [HIDDEN2][HIDDEN1],
    input  logic signed [WIDTH-1:0] B2          [HIDDEN2],
    input  logic signed [WIDTH-1:0] W3          [OUT_SIZE][HIDDEN2],
    input  logic signed [WIDTH-1:0] B3          [OUT_SIZE],

    output logic signed [WIDTH-1:0] softmax_out [BATCH][OUT_SIZE]
);

    genvar b, i;

    // ---- Internal signals ----
    logic signed [WIDTH-1:0] dense1_out  [BATCH][HIDDEN1];
    logic signed [WIDTH-1:0] relu1_out   [BATCH][HIDDEN1];
    logic signed [WIDTH-1:0] dense2_out  [BATCH][HIDDEN2];
    logic signed [WIDTH-1:0] relu2_out   [BATCH][HIDDEN2];
    logic signed [WIDTH-1:0] dense3_out  [BATCH][OUT_SIZE];

    // ---- Layer 1: Dense + Bias ----
    dense_layer #(
        .B(BATCH), .M(HIDDEN1), .N(IN_SIZE), .WIDTH(WIDTH), .FRAC(FRAC)
    ) layer1 (
        .weights(W1),
        .bias(B1),
        .inputs(in_vec),
        .result(dense1_out)
    );

    // ---- Layer 1: ReLU ----
    generate
        for (b = 0; b < BATCH; b++) begin : batch_loop1
            for (i = 0; i < HIDDEN1; i++) begin : neuron_loop1
                relu #(.WIDTH(WIDTH), .FRAC(FRAC)) relu_inst1 (
                    .in(dense1_out[b][i]),
                    .out(relu1_out[b][i])
                );
            end
        end
    endgenerate

    // ---- Layer 2: Dense + Bias ----
    dense_layer #(
        .B(BATCH), .M(HIDDEN2), .N(HIDDEN1), .WIDTH(WIDTH), .FRAC(FRAC)
    ) layer2 (
        .weights(W2),
        .bias(B2),
        .inputs(relu1_out),
        .result(dense2_out)
    );

    // ---- Layer 2: ReLU ----
    generate
        for (b = 0; b < BATCH; b++) begin : batch_loop2
            for (i = 0; i < HIDDEN2; i++) begin : neuron_loop2
                relu #(.WIDTH(WIDTH), .FRAC(FRAC)) relu_inst2 (
                    .in(dense2_out[b][i]),
                    .out(relu2_out[b][i])
                );
            end
        end
    endgenerate

    // ---- Layer 3: Dense + Bias (logits) ----
    dense_layer #(
        .B(BATCH), .M(OUT_SIZE), .N(HIDDEN2), .WIDTH(WIDTH), .FRAC(FRAC)
    ) layer3 (
        .weights(W3),
        .bias(B3),
        .inputs(relu2_out),
        .result(dense3_out)
    );

    // ---- Softmax ----
    generate
        for (b = 0; b < BATCH; b++) begin : batch_softmax
            softmax_fixedpt #(
                .WIDTH(WIDTH),
                .FRAC(FRAC),
                .SIZE(OUT_SIZE)
            ) sm (
                .in(dense3_out[b]),
                .out(softmax_out[b])
            );
        end
    endgenerate

endmodule

