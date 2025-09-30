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
    input  logic signed [WIDTH-1:0] B1         [HIDDEN1],       // Layer1 biases
    input  logic signed [WIDTH-1:0] W2         [OUT_SIZE][HIDDEN1],
    input  logic signed [WIDTH-1:0] B2         [OUT_SIZE],       // Layer2 biases
    output logic signed [WIDTH-1:0] softmax_out[BATCH][OUT_SIZE] // Final softmax
);

    // ---- Internal signals ----
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

    // ---- Layer 3: Softmax ----
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

