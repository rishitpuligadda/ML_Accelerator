`timescale 1ns/1ps

module tb_softmax;

    // Parameters
    localparam WIDTH = 16;
    localparam FRAC  = 8;   // Q8.8
    localparam SIZE  = 4;

    // Signals
    logic signed [WIDTH-1:0] in[SIZE];
    logic signed [WIDTH-1:0] out[SIZE];

    // Instantiate DUT
    softmax_fixedpt #(
        .WIDTH(WIDTH),
        .FRAC(FRAC),
        .SIZE(SIZE)
    ) dut (
        .in(in),
        .out(out)
    );

    // Helper functions for Q8.8 conversion
    function automatic logic signed [WIDTH-1:0] to_fixed(input real val);
        to_fixed = $rtoi(val * (1 << FRAC));
    endfunction

    function automatic real to_real(input logic signed [WIDTH-1:0] val);
        to_real = val / real'(1 << FRAC);
    endfunction

    // Test values
    real test_vectors[0:1][0:SIZE-1]; // two test vectors for demonstration

    initial begin
        // Example test vectors (can expand for more)
        test_vectors[0] = '{1.0, 2.0, 0.5, -1.0};
        test_vectors[1] = '{-1.2, 0.0, 0.5, 2.0};

        $display("Softmax Fixed-Point Testbench");
        $display("Time\tInput(Fixed/Real)\tOutput(Fixed/Real)");

        // Loop over test vectors
        for (int v = 0; v < 2; v = v + 1) begin
            // Apply inputs
            for (int i = 0; i < SIZE; i = i + 1)
                in[i] = to_fixed(test_vectors[v][i]);

            #10; // wait for combinational logic

            // Display outputs
            $display("Vector %0d:", v);
            for (int i = 0; i < SIZE; i = i + 1) begin
                $display("  Input[%0d] = %0d / %f\tOutput[%0d] = %0d / %f",
                         i, in[i], to_real(in[i]), i, out[i], to_real(out[i]));
            end
            $display("");
        end

        #10 $finish;
    end

endmodule

