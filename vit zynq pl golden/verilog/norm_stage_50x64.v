`include "fixed_params.vh"

module norm_stage_50x64 (
    input  signed [`MAT_T_D_W-1:0] x_flat,
    input  signed [`VEC_D_W-1:0]   w_flat,
    input  signed [`VEC_D_W-1:0]   b_flat,
    output reg signed [`MAT_T_D_W-1:0] y_flat
);

    reg signed [`WIDTH-1:0] x  [0:`T-1][0:`D-1];
    reg signed [`WIDTH-1:0] w  [0:`D-1];
    reg signed [`WIDTH-1:0] b  [0:`D-1];
    reg signed [`WIDTH-1:0] dx [0:`D-1];

    integer i, j;
    reg signed [63:0] sum;
    reg signed [63:0] sumsq;
    reg signed [`WIDTH-1:0] mean;
    reg signed [`WIDTH-1:0] var_fixed;
    reg signed [`WIDTH-1:0] std_fixed;
    reg signed [63:0] tmp;

    function automatic [31:0] isqrt;
        input [63:0] x_in;
        reg [63:0] op;
        reg [63:0] res;
        reg [63:0] one;
        begin
            op  = x_in;
            res = 0;
            one = 64'h4000000000000000;
            while (one > op)
                one = one >> 2;
            while (one != 0) begin
                if (op >= res + one) begin
                    op  = op - (res + one);
                    res = res + (one << 1);
                end
                res = res >> 1;
                one = one >> 2;
            end
            isqrt = res[31:0];
        end
    endfunction

    always @(*) begin
        y_flat = {`MAT_T_D_W{1'b0}};

        for (i = 0; i < `T; i = i + 1)
            for (j = 0; j < `D; j = j + 1)
                x[i][j] = x_flat[((i*`D+j)+1)*`WIDTH-1 -: `WIDTH];

        for (j = 0; j < `D; j = j + 1) begin
            w[j] = w_flat[(j+1)*`WIDTH-1 -: `WIDTH];
            b[j] = b_flat[(j+1)*`WIDTH-1 -: `WIDTH];
        end

        for (i = 0; i < `T; i = i + 1) begin
            sum = 0;
            for (j = 0; j < `D; j = j + 1)
                sum = sum + x[i][j];
            mean = sum >>> 6;  // D=64 in the full-scale MNIST block.

            sumsq = 0;
            for (j = 0; j < `D; j = j + 1) begin
                dx[j] = x[i][j] - mean;
                sumsq = sumsq + dx[j] * dx[j];
            end
            var_fixed = ((sumsq >>> 6) >>> `FRAC);
            std_fixed = isqrt((var_fixed + `EPS) <<< `FRAC);

            for (j = 0; j < `D; j = j + 1) begin
                tmp = (dx[j] <<< `FRAC) / std_fixed;
                `ELEM(y_flat, i*`D+j) = ((tmp * w[j]) >>> `FRAC) + b[j];
            end
        end
    end

endmodule
