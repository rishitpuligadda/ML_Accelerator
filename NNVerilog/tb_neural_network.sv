`timescale 1ns/1ps

module tb_neural_network_2layer;

    // ---- Parameters ----
    parameter IN_SIZE    = 2;
    parameter HIDDEN1    = 64;
    parameter OUT_SIZE   = 3;
    parameter WIDTH      = 18;
    parameter FRAC       = 8;
    parameter BATCH      = 300;

    // ---- Signals ----
    logic signed [WIDTH-1:0] in_vec      [BATCH][IN_SIZE];
    logic signed [WIDTH-1:0] W1          [HIDDEN1][IN_SIZE];
    logic signed [WIDTH-1:0] B1          [HIDDEN1];          
    logic signed [WIDTH-1:0] W2          [OUT_SIZE][HIDDEN1];
    logic signed [WIDTH-1:0] B2          [OUT_SIZE];         
    logic signed [WIDTH-1:0] softmax_out [BATCH][OUT_SIZE];
    logic signed [WIDTH-1:0] dense2_dbg  [BATCH][OUT_SIZE];   // Debug Layer 2

    // ---- Instantiate Neural Network ----
    neural_network_2layer_softmax #(
        .IN_SIZE(IN_SIZE),
        .HIDDEN1(HIDDEN1),
        .OUT_SIZE(OUT_SIZE),
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .BATCH(BATCH)
    ) nn (
        .in_vec(in_vec),
        .W1(W1),
        .B1(B1),
        .W2(W2),
        .B2(B2),
        .softmax_out(softmax_out),
        .debug_dense2(dense2_dbg)   // Connect Layer 2 outputs
    );

    // ---- Fixed-point conversion functions ----
    function automatic logic signed [WIDTH-1:0] to_fixed(input real val);
        to_fixed = $rtoi(val * (1 << FRAC));
    endfunction

    function automatic real to_real(input logic signed [WIDTH-1:0] val);
        to_real = val / real'(1 << FRAC);
    endfunction

    initial begin
        int file, fout;
        real temp;
        int i, j;

        // ---- Load Layer 1 weights ----
        file = $fopen("../parameters/weights_layer1.txt", "r");
        if (file == 0) $fatal("Cannot open weights_layer1.txt");
        for (i = 0; i < HIDDEN1; i++)
            for (j = 0; j < IN_SIZE; j++)
                if ($fscanf(file, "%f", temp) != 1)
                    $fatal("Not enough weights in weights_layer1.txt");
                else
                    W1[i][j] = to_fixed(temp);
        $fclose(file);

        // ---- Load Layer 1 biases ----
        file = $fopen("../parameters/biases_layer1.txt", "r");
        if (file == 0) $fatal("Cannot open biases_layer1.txt");
        for (i = 0; i < HIDDEN1; i++) begin
            if ($fscanf(file, "%f", temp) != 1) $fatal("Bad biases_layer1");
            B1[i] = to_fixed(temp);
        end
        $fclose(file);

        // ---- Load Layer 2 weights ----
        file = $fopen("../parameters/weights_layer2.txt", "r");
        if (file == 0) $fatal("Cannot open weights_layer2.txt");
        for (i = 0; i < OUT_SIZE; i++)
            for (j = 0; j < HIDDEN1; j++)
                if ($fscanf(file, "%f", temp) != 1)
                    $fatal("Not enough weights in weights_layer2.txt");
                else
                    W2[i][j] = to_fixed(temp);
        $fclose(file);

        // ---- Load Layer 2 biases ----
        file = $fopen("../parameters/biases_layer2.txt", "r");
        if (file == 0) $fatal("Cannot open biases_layer2.txt");
        for (i = 0; i < OUT_SIZE; i++) begin
            if ($fscanf(file, "%f", temp) != 1) $fatal("Bad biases_layer2");
            B2[i] = to_fixed(temp);
        end
        $fclose(file);

        // ---- Load inputs ----
        file = $fopen("../parameters/inputs.txt", "r");
        if (file == 0) $fatal("Cannot open inputs.txt");
        for (i = 0; i < BATCH; i++)
            for (j = 0; j < IN_SIZE; j++)
                if ($fscanf(file, "%f", temp) != 1)
                    $fatal("Not enough inputs in inputs.txt");
                else
                    in_vec[i][j] = to_fixed(temp);
        $fclose(file);

        #1; // settle combinational outputs

        // ---- Print Layer 2 (Dense) Outputs ----
        for (int b = 0; b < BATCH; b++) begin
            $display("Batch %0d Layer2 (Dense) outputs:", b);
            for (int i = 0; i < OUT_SIZE; i++) begin
                $display("  dense2_dbg[%0d] = %0d (fixed), %f (real)",
                         i, dense2_dbg[b][i], to_real(dense2_dbg[b][i]));
            end
        end

        // ---- Compute softmax max indices and write to file ----
        fout = $fopen("../parameters/outputs.txt", "w");
        if (fout == 0) $fatal("Cannot open outputs.txt for writing");

        for (int b = 0; b < BATCH; b++) begin
            automatic int max_idx = 0;
            automatic logic signed [WIDTH-1:0] max_val = softmax_out[b][0];
            for (int i = 1; i < OUT_SIZE; i++) begin
                if (softmax_out[b][i] > max_val) begin
                    max_val = softmax_out[b][i];
                    max_idx = i;
                end
            end
            $fwrite(fout, "%0d\n", max_idx);

            // Display softmax
            $display("Batch %0d softmax:", b);
            for (int i = 0; i < OUT_SIZE; i++)
                $display("  softmax[%0d] = %f", i, to_real(softmax_out[b][i]));
        end

        $fclose(fout);
        $display("Max indices written to outputs.txt");
        #5 $finish;
    end
endmodule

