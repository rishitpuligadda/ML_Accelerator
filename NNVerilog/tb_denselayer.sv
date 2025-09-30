`timescale 1ns/1ps

module tb_dense_layer;
    parameter B = 2;
    parameter M = 3;
    parameter N = 4;
    parameter WIDTH = 16;
    parameter FRAC  = 8;   // Q8.8 fixed-point

    logic signed [WIDTH-1:0] weights [M][N];
    logic signed [WIDTH-1:0] inputs  [B][N];
    logic signed [WIDTH-1:0] result  [B][M];

    // Instantiate DUT
    dense_layer #(.B(B), .M(M), .N(N), .WIDTH(WIDTH), .FRAC(FRAC)) dut (
        .weights(weights), 
        .inputs(inputs), 
        .result(result)
    );

    // Function to convert real -> fixed-point
    function automatic logic signed [WIDTH-1:0] to_fixed(input real val);
        to_fixed = $rtoi(val * (1 << FRAC));
    endfunction

    // Function to convert fixed-point -> real
    function automatic real to_real(input logic signed [WIDTH-1:0] val);
        to_real = val / real'(1 << FRAC);
    endfunction

    initial begin
        int file;
        real temp;
        int i, j;

        // ---- Load weights from file ----
        file = $fopen("../parameters/weights_layer1.txt", "r");
        if (file == 0) $fatal("Error: could not open weights.txt");
        for (i = 0; i < M; i++) begin
            for (j = 0; j < N; j++) begin
                if ($fscanf(file, "%f", temp) != 1)
                    $fatal("Error: not enough weights in weights.txt");
                weights[i][j] = to_fixed(temp);
            end
        end
        $fclose(file);

        // ---- Load inputs from file ----
        file = $fopen("../parameters/inputs.txt", "r");
        if (file == 0) $fatal("Error: could not open inputs.txt");
        for (i = 0; i < B; i++) begin
            for (j = 0; j < N; j++) begin
                if ($fscanf(file, "%f", temp) != 1)
                    $fatal("Error: not enough inputs in inputs.txt");
                inputs[i][j] = to_fixed(temp);
            end
        end
        $fclose(file);

        #1;

        // ---- Display results ----
        for(int b = 0; b < B; b++) begin
            $display("Batch[%0d] Inputs (fixed) = %p", b, inputs[b]);
            for(int i = 0; i < M; i++) begin
                $display("  Neuron[%0d]: Output = %0d (fixed), %f (real)", 
                          i, result[b][i], to_real(result[b][i]));
            end
        end

        #5 $finish;
    end
endmodule

