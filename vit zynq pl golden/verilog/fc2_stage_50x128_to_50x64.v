`include "fixed_params.vh"

module fc2_stage_50x128_to_50x64 (
    input  signed [`MAT_T_DFF_W-1:0] x_flat,
    input  signed [`W_D_DFF_W-1:0]   w_flat,   // [D][DFF], row-major, out-first
    input  signed [`VEC_D_W-1:0]     b_flat,
    output reg signed [`MAT_T_D_W-1:0] y_flat
);

    reg signed [`WIDTH-1:0] x [0:`T-1][0:`DFF-1];
    reg signed [`WIDTH-1:0] w [0:`D-1][0:`DFF-1];
    reg signed [`WIDTH-1:0] b [0:`D-1];
    reg signed [`WIDTH-1:0] y [0:`T-1][0:`D-1];

    integer i, j, d;
    reg signed [63:0] acc;

    always @(*) begin
        y_flat = {`MAT_T_D_W{1'b0}};

        for (i = 0; i < `T; i = i + 1)
            for (d = 0; d < `DFF; d = d + 1)
                x[i][d] = x_flat[((i*`DFF+d)+1)*`WIDTH-1 -: `WIDTH];

        for (j = 0; j < `D; j = j + 1)
            for (d = 0; d < `DFF; d = d + 1)
                w[j][d] = w_flat[((j*`DFF+d)+1)*`WIDTH-1 -: `WIDTH];

        for (j = 0; j < `D; j = j + 1)
            b[j] = b_flat[(j+1)*`WIDTH-1 -: `WIDTH];

        for (i = 0; i < `T; i = i + 1) begin
            for (j = 0; j < `D; j = j + 1) begin
                acc = 0;
                for (d = 0; d < `DFF; d = d + 1)
                    acc = acc + x[i][d] * w[j][d];
                acc = acc + ({{48{b[j][`WIDTH-1]}}, b[j]} <<< `FRAC);
                y[i][j] = acc >>> `FRAC;
            end
        end

        for (i = 0; i < `T; i = i + 1)
            for (j = 0; j < `D; j = j + 1)
                y_flat[((i*`D+j)+1)*`WIDTH-1 -: `WIDTH] = y[i][j];
    end

endmodule
