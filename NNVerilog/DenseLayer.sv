module dense_layer #(
    parameter B = 2,                  // batch size
    parameter M = 3,                  // number of neurons
    parameter N = 4,                  // input features
    parameter WIDTH = 16,             // bit width
    parameter FRAC = 8                // number of fractional bits (Qm.n)
)(
    input  logic signed [WIDTH-1:0] weights [M][N],   // Q format
    input  logic signed [WIDTH-1:0] inputs  [B][N],   // Q format
    output logic signed [WIDTH-1:0] result  [B][M]    // Q format
);

    integer b, i, j;
    logic signed [2*WIDTH-1:0] sum;
    logic signed [2*WIDTH-1:0] product;

    always_comb begin
        for(b = 0; b < B; b++) begin
            for(i = 0; i < M; i++) begin
                sum = 0;
                for(j = 0; j < N; j++) begin
                    product = weights[i][j] * inputs[b][j];  // Q(FRAC*2)
                    sum += product;
                end
                // shift right to return to Q(FRAC)
                result[b][i] = sum >>> FRAC;
            end
        end
    end
endmodule
