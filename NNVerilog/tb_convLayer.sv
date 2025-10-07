`timescale 1ns/1ps

module tb_convLayer;

  // Parameters for dilated 3x3 conv
  localparam int D_IN       = 1;     // input channels
  localparam int H_IN       = 5;
  localparam int W_IN       = 5;
  localparam int N_FILTERS  = 64;    // output channels
  localparam int H_FILT     = 3;     // 3x3 kernel
  localparam int W_FILT     = 3;
  localparam int STRIDE     = 1;
  localparam int PADDING    = 2;     // for dilation=2 and "same" output
  localparam int DILATION   = 2;     // dilation factor
  localparam int DATA_WIDTH = 18;
  localparam int FRAC_WIDTH = 8;     // Q8.8 fixed-point

  // DUT signals
  logic clk;
  logic rst_n;
  logic start;
  logic done;

  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = ~clk;
  end

  // Instantiate DUT
  conv_forward_inference_fixed #(
    .D_IN(D_IN),
    .H_IN(H_IN),
    .W_IN(W_IN),
    .N_FILTERS(N_FILTERS),
    .H_FILT(H_FILT),
    .W_FILT(W_FILT),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .DILATION(DILATION),
    .DATA_WIDTH(DATA_WIDTH),
    .FRAC_WIDTH(FRAC_WIDTH)
  ) dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .done(done)
  );

  // Initialize DUT memory
  initial begin
    localparam int SCALE = 256; // Q8.8 scaling
    rst_n = 0;
    start = 0;
    #10;
    rst_n = 1;

    // Custom input feature map (summed to 1 channel)
    dut.X[0] = '{
      '{ 6*SCALE, 3*SCALE, 2*SCALE, 3*SCALE, 6*SCALE },
      '{ 1*SCALE, 2*SCALE, 1*SCALE, 2*SCALE, 1*SCALE },
      '{ 2*SCALE, 1*SCALE, 2*SCALE, 1*SCALE, 2*SCALE },
      '{ 5*SCALE, 4*SCALE, 5*SCALE, 4*SCALE, 5*SCALE },
      '{ 1*SCALE, 2*SCALE, 1*SCALE, 2*SCALE, 1*SCALE }
    };

    // Initialize 3x3 kernel weights for all 64 filters
    for (int f = 0; f < N_FILTERS; f++) begin
      for (int c = 0; c < D_IN; c++) begin
        for (int i = 0; i < H_FILT; i++) begin
          for (int j = 0; j < W_FILT; j++) begin
            dut.Wght[f][c][i][j] = 1*SCALE; // all ones for simplicity
          end
        end
      end
      dut.Bias[f] = 0;
    end

    // Start convolution
    #10;
    start = 1;
    #10;
    start = 0;

    // Wait for done
    wait(done);

    // Display output for all filters with ReLU applied
    $display("Dilated Convolution Output (all filters, floating point, ReLU):");
    for (int f = 0; f < N_FILTERS; f++) begin
        $display("Filter %0d:", f);
        for (int i = 0; i < H_IN; i++) begin
            for (int j = 0; j < W_IN; j++) begin
                automatic real val;
                val = dut.Y[f][i][j] / (2.0 ** FRAC_WIDTH);
                // Apply ReLU
                if (val < 0) val = 0;
                $write("%0f ", val);
            end
            $write("\n");
        end
        $write("\n");
    end

    $finish;
  end

endmodule

