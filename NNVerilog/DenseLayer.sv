module dense_layer #(
    parameter B = 2,
    parameter M = 3,
    parameter N = 4,
    parameter WIDTH = 16
)(
    input logic signed [WIDTH-1:0] weights [M][N],
    input logic signed [WIDTH-1:0] inputs [B][N],
    output logic signed [2*WIDTH-1:0] result [B][M]
);

    integer b, i, j;
    logic signed [2*WIDTH-1:0] sum;

    always_comb begin
        for(b = 0; b < B; b++) begin
            for(i = 0; i < M; i++) begin
                sum = 0;
                for(j = 0; j < N; j++) begin
                    sum += weights[i][j] * inputs[b][j];
                end
                result[b][i] = sum;
            end
        end
    end
endmodule
