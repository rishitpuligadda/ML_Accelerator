`timescale 1ns/1ps

module tb_relu;

  // Parameters
  localparam WIDTH = 16;
  localparam FRAC  = 8;   // Q8.8 format

  // Signals
  logic signed [WIDTH-1:0] in;
  logic signed [WIDTH-1:0] out;

  // Instantiate DUT
  relu #(.WIDTH(WIDTH), .FRAC(FRAC)) dut (
    .in(in),
    .out(out)
  );

  // Helper functions
  function automatic logic signed [WIDTH-1:0] to_fixed(input real val);
    to_fixed = $rtoi(val * (1 << FRAC));
  endfunction

  function automatic real to_real(input logic signed [WIDTH-1:0] val);
    to_real = val / real'(1 << FRAC);
  endfunction

  // Stimulus
  initial begin
   // Declare variables at the top of the block
   real test_values[0:2];
   integer i;

   $display("Time\tInput(fixed/real)\tOutput(fixed/real)");

    // Initialize array inside the block
    test_values[0] = -1.19751198;
    test_values[1] = 0.012887491;
    test_values[2] = 0.42328643;

    // Apply test values
    for(i = 0; i < 3; i = i + 1) begin
        in = to_fixed(test_values[i]);
        #10;
        $display("%0t\t%d / %f\t%d / %f", $time, in, to_real(in), out, to_real(out));
    end

    #10 $finish;
  end

endmodule

