module Convolutional #(
    parameter int IN_DEPTH    = 3,
    parameter int IN_HEIGHT   = 4,
    parameter int IN_WIDTH    = 4,
    parameter int OUT_DEPTH   = 2,
    parameter int KERNEL_SIZE = 3,
    parameter int DATA_W      = 16,
    parameter int FRAC        = 4   // fractional bits for fixed-point
)(
    // Input: [in_depth][H][W]  (single batch)
    input  logic signed [DATA_W-1:0] input_data  [0:IN_DEPTH-1][0:IN_HEIGHT-1][0:IN_WIDTH-1],

    // Kernels: [kernel_h][kernel_w][in_depth][out_depth]
    input  logic signed [DATA_W-1:0] kernels     [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1][0:IN_DEPTH-1][0:OUT_DEPTH-1],

    // Bias: [out_depth] — scalar per output channel
    input  logic signed [DATA_W-1:0] biases      [0:OUT_DEPTH-1],

    // Output: [out_depth][H][W]
    output logic signed [DATA_W-1:0] output_data [0:OUT_DEPTH-1][0:IN_HEIGHT-1][0:IN_WIDTH-1]
);

    integer r, c, cout, cin, m, n;
    logic signed [2*DATA_W-1:0] acc; 
    int pad_top, pad_left;

    always_comb begin
        pad_top  = KERNEL_SIZE / 2;
        pad_left = KERNEL_SIZE / 2;

        // Loop over output channels (filters)
        for (cout = 0; cout < OUT_DEPTH; cout++) begin
            for (r = 0; r < IN_HEIGHT; r++) begin
                for (c = 0; c < IN_WIDTH; c++) begin
                    // Start accumulation with scalar bias
                    acc = biases[cout];

                    // Loop over input channels
                    for (cin = 0; cin < IN_DEPTH; cin++) begin
                        // Loop over kernel spatial dims
                        for (m = 0; m < KERNEL_SIZE; m++) begin
                            for (n = 0; n < KERNEL_SIZE; n++) begin
                                automatic int in_r = r + m - pad_top;
                                automatic int in_c = c + n - pad_left;

                                if (in_r >= 0 && in_r < IN_HEIGHT &&
                                    in_c >= 0 && in_c < IN_WIDTH) begin
                                    acc += input_data[cin][in_r][in_c] *
                                           kernels[m][n][cin][cout];
                                end
                            end
                        end
                    end

                    // Fixed-point shift
                    output_data[cout][r][c] = acc >>> FRAC;
                end
            end
        end
    end

endmodule

