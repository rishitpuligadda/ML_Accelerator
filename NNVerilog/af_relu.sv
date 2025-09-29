module relu #(
    parameter WIDTH = 16, 
    parameter FRAC  = 8 
)(
    input  logic signed [WIDTH-1:0] in,
    output logic signed [WIDTH-1:0] out 
);

    always_comb begin
        if (in > 0)
            out = in;
        else
            out = '0;
    end

endmodule
