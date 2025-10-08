`timescale 1ns/1ps

module tb_convolutional_fp_32;

    // ---- Parameters ----
    parameter int IN_DEPTH    = 2;
    parameter int IN_HEIGHT   = 5;
    parameter int IN_WIDTH    = 5;
    parameter int OUT_DEPTH   = 32;
    parameter int KERNEL_SIZE = 3;
    parameter int DATA_W      = 18;
    parameter int FRAC        = 9; // fractional bits for fixed-point

    // ---- Signals ----
    logic signed [DATA_W-1:0] input_data  [0:IN_DEPTH-1][0:IN_HEIGHT-1][0:IN_WIDTH-1];
    logic signed [DATA_W-1:0] kernels     [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1][0:IN_DEPTH-1][0:OUT_DEPTH-1];
    logic signed [DATA_W-1:0] biases      [0:OUT_DEPTH-1];
    logic signed [DATA_W-1:0] output_data [0:OUT_DEPTH-1][0:IN_HEIGHT-1][0:IN_WIDTH-1];

    // ---- Convolution Instance ----
    Convolutional #(
        .IN_DEPTH(IN_DEPTH),
        .IN_HEIGHT(IN_HEIGHT),
        .IN_WIDTH(IN_WIDTH),
        .OUT_DEPTH(OUT_DEPTH),
        .KERNEL_SIZE(KERNEL_SIZE),
        .DATA_W(DATA_W),
        .FRAC(FRAC)
    ) conv_inst (
        .input_data(input_data),
        .kernels(kernels),
        .biases(biases),
        .output_data(output_data)
    );

    integer i, r, c, cin, cout, kh, kw;

    // ---- Fixed-point conversion function ----
    function automatic logic signed [DATA_W-1:0] to_fixed(input real val);
        to_fixed = $rtoi(val * (1 << FRAC));
    endfunction

    initial begin
        // ---- Initialize input_data ----
        input_data[0] = '{'{to_fixed( 1.5), to_fixed( 2.1), to_fixed(-0.5), to_fixed( 4.2), to_fixed( 5.7)},
                          '{to_fixed(-6.5), to_fixed( 7.1), to_fixed( 8.8), to_fixed(-9.5), to_fixed(10.3)},
                          '{to_fixed(11.2), to_fixed(-12.3), to_fixed(13.8), to_fixed(14.6), to_fixed(-15.1)},
                          '{to_fixed(16.3), to_fixed(-17.4), to_fixed(18.5), to_fixed(19.6), to_fixed(20.2)},
                          '{to_fixed(-21.5), to_fixed(22.4), to_fixed(23.7), to_fixed(-24.8), to_fixed(25.9)}};

        input_data[1] = '{'{to_fixed(-3.2), to_fixed(4.8), to_fixed(1.3), to_fixed(-2.1), to_fixed(0.0)},
                          '{to_fixed(10.2), to_fixed(-9.3), to_fixed(8.1), to_fixed(7.5), to_fixed(-6.6)},
                          '{to_fixed(-15.5), to_fixed(14.7), to_fixed(-13.2), to_fixed(12.4), to_fixed(11.9)},
                          '{to_fixed(20.7), to_fixed(19.1), to_fixed(-18.2), to_fixed(17.8), to_fixed(-16.5)},
                          '{to_fixed(25.3), to_fixed(-24.6), to_fixed(23.9), to_fixed(22.2), to_fixed(-21.1)}};

        // ---- Initialize kernels deterministically ----
        for (cout=0; cout<OUT_DEPTH; cout++) begin
            for (cin=0; cin<IN_DEPTH; cin++) begin
                for (kh=0; kh<KERNEL_SIZE; kh++) begin
                    for (kw=0; kw<KERNEL_SIZE; kw++) begin
                        kernels[kh][kw][cin][cout] = to_fixed( ((kh+1)*(kw+1) + cin + cout)*0.01 );
                    end
                end
            end
        end

        // ---- Initialize biases to zero ----
        for (cout=0; cout<OUT_DEPTH; cout++) begin
            biases[cout] = to_fixed(0.0);
        end

        // Wait a little for combinational logic to settle
        #1;

        // ---- Display first 2 channels as sanity check ----
        for (i = 0; i < 32; i++) begin
            $display("Output channel %0d:", i);
            for (r = 0; r < IN_HEIGHT; r++) begin
                for (c = 0; c < IN_WIDTH; c++) begin
                    $write("%0.4f ", $itor(output_data[i][r][c]) / (1 << FRAC));
                end
                $write("\n");
            end
            $write("\n");
        end

        $finish;
    end

endmodule

