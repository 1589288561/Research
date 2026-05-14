`include "fixed_params.vh"

module residual_add_50x64 (
    input  signed [`MAT_T_D_W-1:0] a_flat,
    input  signed [`MAT_T_D_W-1:0] b_flat,
    output signed [`MAT_T_D_W-1:0] y_flat
);
    genvar i;
    generate
        for (i = 0; i < `T*`D; i = i + 1) begin : G_ADD
            assign `ELEM(y_flat, i) = `ELEM(a_flat, i) + `ELEM(b_flat, i);
        end
    endgenerate
endmodule
