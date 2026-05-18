`include "fixed_params.vh"

module gelu_stage_50x128 (
    input  signed [`MAT_T_DFF_W-1:0] x_flat,
    output reg signed [`MAT_T_DFF_W-1:0] y_flat
);

    `include "gelu_lut_q412_full.vh"

    integer i;

    always @(*) begin
        y_flat = {`MAT_T_DFF_W{1'b0}};

        for (i = 0; i < `T*`DFF; i = i + 1) begin
            `ELEM(y_flat, i) = gelu_lut_lookup($signed(`ELEM(x_flat, i)));
        end
    end

endmodule
