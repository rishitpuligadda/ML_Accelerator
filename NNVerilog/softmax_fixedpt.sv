`timescale 1ns/1ps

module exp_approx #(
    parameter WIDTH = 16,
    parameter FRAC  = 8
)(
    input  logic signed [WIDTH-1:0] x,
    output logic signed [WIDTH-1:0] y
);
    localparam signed [WIDTH-1:0] MIN_X = -(2 <<< FRAC); 
    logic signed [WIDTH-1:0] x_clip;
    logic signed [31:0] x32, term2, term3, term4, term5, term6;
    logic signed [31:0] result;

    always_comb begin
        if (x < MIN_X)
            x_clip = MIN_X;
        else
            x_clip = x;

        x32 = x_clip;

        term2 = (x32 * x32) >>> FRAC; term2 = term2 / 2;
        term3 = (term2 * x32) >>> FRAC; term3 = term3 / 3;
        term4 = (term3 * x32) >>> FRAC; term4 = term4 / 4;
        term5 = (term4 * x32) >>> FRAC; term5 = term5 / 5;
        term6 = (term5 * x32) >>> FRAC; term6 = term6 / 6;

        result = (1 <<< FRAC) + x32 + term2 + term3 + term4 + term5 + term6;

        if (result < 0)
            result = 0;

        y = result[WIDTH-1:0];
    end
endmodule

module softmax_fixedpt #(
    parameter WIDTH = 16,
    parameter FRAC  = 8,
    parameter SIZE  = 4
)(
    input  logic signed [WIDTH-1:0] in [SIZE],
    output logic signed [WIDTH-1:0] out [SIZE]
);

    logic signed [WIDTH-1:0] max_val;
    logic signed [WIDTH-1:0] sub_val[SIZE];
    logic signed [WIDTH-1:0] exp_val[SIZE];
    logic signed [31:0] sum_exp;

    // Step 1: Find max input
    always_comb begin
        int j;
        max_val = in[0];
        for (j = 1; j < SIZE; j++)
            if (in[j] > max_val)
                max_val = in[j];
    end

    // Step 2: Subtract max to avoid overflow
    genvar gi;
    generate
        for (gi = 0; gi < SIZE; gi++)
            assign sub_val[gi] = in[gi] - max_val;
    endgenerate

    // Step 3: Compute exponential for each element
    genvar ge;
    generate
        for (ge = 0; ge < SIZE; ge++)
            exp_approx #(.WIDTH(WIDTH), .FRAC(FRAC)) exp_i (
                .x(sub_val[ge]),
                .y(exp_val[ge])
            );
    endgenerate

    // Step 4: Sum exponentials
    always_comb begin
        int j;
        sum_exp = 0;
        for (j = 0; j < SIZE; j++)
            sum_exp += exp_val[j];
        if (sum_exp < 0)
            sum_exp = 0;
    end

    // Step 5: Normalize with rounding
    logic signed [WIDTH-1:0] temp_out[SIZE];
    always_comb begin
        int j;
        for (j = 0; j < SIZE; j++) begin
            logic signed [31:0] numerator;
            numerator = ($signed(exp_val[j]) <<< FRAC) + (sum_exp >> 1); // rounding
            if (sum_exp == 0)
                temp_out[j] = '0;
            else
                temp_out[j] = $signed(numerator / sum_exp);
        end
    end

    // Step 6: Adjust largest element so sum = 256
    always_comb begin
        int j, max_idx;
        int signed sum_temp;
        sum_temp = 0;
        max_idx = 0;

        // Find the max element in temp_out
        for (j = 0; j < SIZE; j = j + 1) begin
            sum_temp += temp_out[j];
            if (temp_out[j] > temp_out[max_idx])
                max_idx = j;
        end

        // Assign final outputs
        for (j = 0; j < SIZE; j = j + 1)
            out[j] = temp_out[j];

        // Adjust largest element to make sum exactly 256
        out[max_idx] = out[max_idx] + (256 - sum_temp);
    end
endmodule
