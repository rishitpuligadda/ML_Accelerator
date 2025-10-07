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

module tb_neural_network_3layer_single;

    // ---- Parameters ----
    parameter IN_SIZE    = 784;
    parameter HIDDEN1    = 128;
    parameter HIDDEN2    = 128;
    parameter OUT_SIZE   = 10;
    parameter WIDTH      = 18;
    parameter FRAC       = 8;
    parameter BATCH      = 1;

    // ---- Signals ----
    logic signed [WIDTH-1:0] in_vec      [BATCH][IN_SIZE];
    logic signed [WIDTH-1:0] W1_fixed    [HIDDEN1][IN_SIZE];
    logic signed [WIDTH-1:0] B1_fixed    [HIDDEN1];
    logic signed [WIDTH-1:0] W2_fixed    [HIDDEN2][HIDDEN1];
    logic signed [WIDTH-1:0] B2_fixed    [HIDDEN2];
    logic signed [WIDTH-1:0] W3_fixed    [OUT_SIZE][HIDDEN2];
    logic signed [WIDTH-1:0] B3_fixed    [OUT_SIZE];
    logic signed [WIDTH-1:0] softmax_out [BATCH][OUT_SIZE];

    // ---- Neural Network Instance ----
    neural_network_3layer_softmax #(
        .IN_SIZE(IN_SIZE),
        .HIDDEN1(HIDDEN1),
        .HIDDEN2(HIDDEN2),
        .OUT_SIZE(OUT_SIZE),
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .BATCH(BATCH)
    ) nn (
        .in_vec(in_vec),
        .W1(W1_fixed),
        .B1(B1_fixed),
        .W2(W2_fixed),
        .B2(B2_fixed),
        .W3(W3_fixed),
        .B3(B3_fixed),
        .softmax_out(softmax_out)
    );

    // ---- Fixed-point conversion function ----
    function automatic logic signed [WIDTH-1:0] to_fixed(input real val);
        to_fixed = $rtoi(val * (1 << FRAC));
    endfunction

    initial begin
        int i, j, f;
        int max_idx;
        logic signed [WIDTH-1:0] max_val;
        real temp;

        // ---- Load weights and biases ----
        static string files[6] = '{
            "../parameters/weights_layer1.txt",
            "../parameters/biases_layer1.txt",
            "../parameters/weights_layer2.txt",
            "../parameters/biases_layer2.txt",
            "../parameters/weights_layer3.txt",
            "../parameters/biases_layer3.txt"
        };

        static string class_labels [OUT_SIZE] = '{
            "T-Shirt",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot"
            };

        // ---- Layer 1 ----
        f = $fopen(files[0], "r");
        if (f == 0) $fatal("Cannot open %s", files[0]);
        for(i=0;i<HIDDEN1;i=i+1)
            for(j=0;j<IN_SIZE;j=j+1) begin
                void'($fscanf(f,"%f", temp));
                W1_fixed[i][j] = to_fixed(temp);
            end
        $fclose(f);

        f = $fopen(files[1], "r");
        if (f == 0) $fatal("Cannot open %s", files[1]);
        for(i=0;i<HIDDEN1;i=i+1) begin
            void'($fscanf(f,"%f", temp));
            B1_fixed[i] = to_fixed(temp);
        end
        $fclose(f);

        // ---- Layer 2 ----
        f = $fopen(files[2], "r");
        if (f == 0) $fatal("Cannot open %s", files[2]);
        for(i=0;i<HIDDEN2;i=i+1)
            for(j=0;j<HIDDEN1;j=j+1) begin
                void'($fscanf(f,"%f", temp));
                W2_fixed[i][j] = to_fixed(temp);
            end
        $fclose(f);

        f = $fopen(files[3], "r");
        if (f == 0) $fatal("Cannot open %s", files[3]);
        for(i=0;i<HIDDEN2;i=i+1) begin
            void'($fscanf(f,"%f", temp));
            B2_fixed[i] = to_fixed(temp);
        end
        $fclose(f);

        // ---- Layer 3 ----
        f = $fopen(files[4], "r");
        if (f == 0) $fatal("Cannot open %s", files[4]);
        for(i=0;i<OUT_SIZE;i=i+1)
            for(j=0;j<HIDDEN2;j=j+1) begin
                void'($fscanf(f,"%f", temp));
                W3_fixed[i][j] = to_fixed(temp);
            end
        $fclose(f);

        f = $fopen(files[5], "r");
        if (f == 0) $fatal("Cannot open %s", files[5]);
        for(i=0;i<OUT_SIZE;i=i+1) begin
            void'($fscanf(f,"%f", temp));
            B3_fixed[i] = to_fixed(temp);
        end
        $fclose(f);

        // ---- Load input image ----
        f = $fopen("../parameters/input_image.txt", "r");
        if(f == 0) $fatal("Cannot open input_image.txt");
        for(i=0;i<IN_SIZE;i=i+1) begin
            void'($fscanf(f,"%f", temp));
            in_vec[0][i] = to_fixed(temp);
        end
        $fclose(f);

        #1; // settle combinational outputs

        // ---- Compute predicted class ----
        max_idx = 0;
        max_val = softmax_out[0][0];
        for(i=1;i<OUT_SIZE;i=i+1) begin
            if(softmax_out[0][i] > max_val) begin
                max_val = softmax_out[0][i];
                max_idx = i;
            end
        end
        $display("Predicted class: %0d (%s)", max_idx, class_labels[max_idx]);

        #5 $finish;
    end
endmodule

