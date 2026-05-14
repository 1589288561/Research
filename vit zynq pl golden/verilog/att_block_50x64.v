`include "fixed_params.vh"

module att_block_50x64 (
    input  signed [`MAT5X4_W-1:0] x_flat,
    input  signed [`VEC4_W-1:0]   n1w_flat,
    input  signed [`VEC4_W-1:0]   n1b_flat,

    input  signed [`W4X4_W-1:0]   wq_flat,
    input  signed [`VEC4_W-1:0]   bq_flat,
    input  signed [`W4X4_W-1:0]   wk_flat,
    input  signed [`VEC4_W-1:0]   bk_flat,
    input  signed [`W4X4_W-1:0]   wv_flat,
    input  signed [`VEC4_W-1:0]   bv_flat,

    output signed [`MAT5X4_W-1:0] norm1_flat,
    output signed [`MAT5X4_W-1:0] att_only_flat,
    output signed [`MAT5X4_W-1:0] y_flat
);

    norm_stage_50x64 u_norm1 (
        .x_flat(x_flat),
        .w_flat(n1w_flat),
        .b_flat(n1b_flat),
        .y_flat(norm1_flat)
    );

    att_core_50x64 u_att_core (
        .x_flat(norm1_flat),
        .wq_flat(wq_flat), .bq_flat(bq_flat),
        .wk_flat(wk_flat), .bk_flat(bk_flat),
        .wv_flat(wv_flat), .bv_flat(bv_flat),
        .ctx_flat(att_only_flat)
    );

    residual_add_50x64 u_res1 (
        .a_flat(x_flat),
        .b_flat(att_only_flat),
        .y_flat(y_flat)
    );

endmodule
