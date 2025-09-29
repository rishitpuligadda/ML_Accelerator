`timescale 1ns/1ps

module neural_network_2layer #(
    parameter IN_SIZE  = 4,
    parameter HIDDEN1  = 3,
    parameter OUT_SIZE = 2,
    parameter WIDTH    = 16,
    parameter FRAC     = 12
)(
    input  logic signed [WIDTH-1:0] in_vec  [1][IN_SIZE],   // batch=1
    input  logic signed [WIDTH-1:0] W1      [HIDDEN1][IN_SIZE],
    input  logic signed [WIDTH-1:0] W2      [OUT_SIZE][HIDDEN1],
    output logic signed [WIDTH-1:0] out_vec [1][OUT_SIZE]
);

    // ---- Layer 1 ----
    logic signed [WIDTH-1:0] hidden1 [1][HIDDEN1];
    logic signed [WIDTH-1:0] relu1   [1][HIDDEN1];

    dense_layer #(.B(1), .M(HIDDEN1), .N(IN_SIZE), .WIDTH(WIDTH), .FRAC(FRAC)) L1 (
        .weights(W1),
        .inputs (in_vec),
        .result (hidden1)
    );

    genvar i;
    generate
        for(i = 0; i < HIDDEN1; i=i+1) begin : relu_layer1
            relu #(.WIDTH(WIDTH), .FRAC(FRAC)) relu1_inst (
                .in (hidden1[0][i]),
                .out(relu1[0][i])
            );
        end
    endgenerate

    // ---- Layer 2 (Output) ----
    logic signed [WIDTH-1:0] hidden2 [1][OUT_SIZE];

    dense_layer #(.B(1), .M(OUT_SIZE), .N(HIDDEN1), .WIDTH(WIDTH), .FRAC(FRAC)) L2 (
        .weights(W2),
        .inputs (relu1),
        .result (hidden2)
    );

    // ---- Softmax ----
    logic signed [WIDTH-1:0] logits [OUT_SIZE];

    genvar j;
    generate
        for(j = 0; j < OUT_SIZE; j=j+1) begin
            assign logits[j] = hidden2[0][j];
        end
    endgenerate

    softmax_fixedpt #(.WIDTH(WIDTH), .FRAC(FRAC), .SIZE(OUT_SIZE)) softmax_inst (
        .in (logits),
        .out(out_vec[0])
    );

endmodule

