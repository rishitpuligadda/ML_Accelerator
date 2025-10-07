module conv_forward_inference_fixed #(
    parameter int D_IN        = 3,    // Number of input channels
    parameter int H_IN        = 64,   // Input height
    parameter int W_IN        = 64,   // Input width
    parameter int N_FILTERS   = 32,   // Number of filters
    parameter int H_FILT      = 3,    // Filter height
    parameter int W_FILT      = 3,    // Filter width
    parameter int STRIDE      = 1,
    parameter int PADDING     = 1,
    parameter int DILATION    = 1,    // New: dilation rate
    parameter int DATA_WIDTH  = 16,   // Total bits
    parameter int FRAC_WIDTH  = 8     // Fractional bits
)(
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    output logic done
);

    // Effective filter size due to dilation
    localparam int H_EFF = H_FILT + (H_FILT-1)*(DILATION-1);
    localparam int W_EFF = W_FILT + (W_FILT-1)*(DILATION-1);

    // Derived output size
    localparam int H_OUT = ((H_IN - H_EFF + 2*PADDING) / STRIDE) + 1;
    localparam int W_OUT = ((W_IN - W_EFF + 2*PADDING) / STRIDE) + 1;

    // Typedef for fixed point
    typedef logic signed [DATA_WIDTH-1:0] fixed_t;

    // Memory arrays
    fixed_t X      [D_IN][H_IN][W_IN];                  // Input feature map
    fixed_t Wght   [N_FILTERS][D_IN][H_FILT][W_FILT];   // Filters
    fixed_t Bias   [N_FILTERS];                         // Biases
    fixed_t Y      [N_FILTERS][H_OUT][W_OUT];           // Output feature map

    // Internal vars
    int f, c, i, j, m, n;
    logic signed [2*DATA_WIDTH-1:0] mult_result;  // double width for multiplication
    logic signed [2*DATA_WIDTH-1:0] acc;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
        end else if (start) begin
            // Loop over filters
            for (f = 0; f < N_FILTERS; f++) begin
                // Loop over output height
                for (i = 0; i < H_OUT; i++) begin
                    // Loop over output width
                    for (j = 0; j < W_OUT; j++) begin
                        // Initialize accumulator with sign-extended bias
                        acc = {{(DATA_WIDTH){Bias[f][DATA_WIDTH-1]}}, Bias[f]};

                        // Convolution window with dilation
                        for (c = 0; c < D_IN; c++) begin
                            for (m = 0; m < H_FILT; m++) begin
                                for (n = 0; n < W_FILT; n++) begin
                                    automatic int in_y;
                                    automatic int in_x;

                                    // Apply dilation here
                                    in_y = i*STRIDE + m*DILATION - PADDING;
                                    in_x = j*STRIDE + n*DILATION - PADDING;

                                    // Bounds check
                                    if (in_y >= 0 && in_y < H_IN &&
                                        in_x >= 0 && in_x < W_IN) begin
                                        // Multiply in fixed point
                                        mult_result = X[c][in_y][in_x] * Wght[f][c][m][n];
                                        // Shift back to maintain Q format
                                        acc += mult_result >>> FRAC_WIDTH;
                                    end
                                end
                            end
                        end

                        // Store result (with saturation if needed)
                        Y[f][i][j] = acc[DATA_WIDTH-1:0];
                    end
                end
            end

            done <= 1'b1;
        end
    end

endmodule

