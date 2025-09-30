module dense_layer #(
    parameter B = 2,
    parameter M = 3,
    parameter N = 4,
    parameter WIDTH = 32,
    parameter FRAC  = 16
)(
    input  logic signed [WIDTH-1:0] weights [M][N],
    input  logic signed [WIDTH-1:0] inputs  [B][N],
    input  logic signed [WIDTH-1:0] bias    [M],
    output logic signed [WIDTH-1:0] result  [B][M]
);

    integer b, i, j;
    logic signed [2*WIDTH-1:0] sum;
    logic signed [2*WIDTH-1:0] product;

    always_comb begin
        for(b = 0; b < B; b++) begin
            for(i = 0; i < M; i++) begin
                sum = 0;
                for(j = 0; j < N; j++) begin
                    product = $signed(weights[i][j]) * $signed(inputs[b][j]); 
                    sum += product;
                end
                sum += $signed(bias[i]) <<< FRAC;  // align bias
                result[b][i] = sum >>> FRAC;       // back to Q(FRAC)
            end
        end
    end
endmodule

