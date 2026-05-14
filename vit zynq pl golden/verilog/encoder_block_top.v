`include "fixed_params.vh"

module encoder_block_top (
    input  signed [`MAT5X4_W-1:0] x_flat,

    input  signed [`VEC4_W-1:0] n1w_flat,
    input  signed [`VEC4_W-1:0] n1b_flat,
    input  signed [`W4X4_W-1:0] wq_flat,
    input  signed [`VEC4_W-1:0] bq_flat,
    input  signed [`W4X4_W-1:0] wk_flat,
    input  signed [`VEC4_W-1:0] bk_flat,
    input  signed [`W4X4_W-1:0] wv_flat,
    input  signed [`VEC4_W-1:0] bv_flat,

    input  signed [`VEC4_W-1:0] n2w_flat,
    input  signed [`VEC4_W-1:0] n2b_flat,
    input  signed [`W8X4_W-1:0] fc1_w_flat,
    input  signed [`VEC8_W-1:0] fc1_b_flat,
    input  signed [`W4X8_W-1:0] fc2_w_flat,
    input  signed [`VEC4_W-1:0] fc2_b_flat,

    output signed [`MAT5X4_W-1:0] norm1_flat,
    output signed [`MAT5X4_W-1:0] att_only_flat,
    output signed [`MAT5X4_W-1:0] att_out_flat,
    output signed [`MAT5X4_W-1:0] norm2_flat,
    output signed [`MAT5X8_W-1:0] fc1_flat,
    output signed [`MAT5X8_W-1:0] gelu_flat,
    output signed [`MAT5X4_W-1:0] fc2_flat,
    output signed [`MAT5X4_W-1:0] y_flat
);

    att_block_50x64 u_att (
        .x_flat(x_flat),
        .n1w_flat(n1w_flat),
        .n1b_flat(n1b_flat),
        .wq_flat(wq_flat), .bq_flat(bq_flat),
        .wk_flat(wk_flat), .bk_flat(bk_flat),
        .wv_flat(wv_flat), .bv_flat(bv_flat),
        .norm1_flat(norm1_flat),
        .att_only_flat(att_only_flat),
        .y_flat(att_out_flat)
    );

    mlp_block_50x64 u_mlp (
        .x_flat(att_out_flat),
        .n2w_flat(n2w_flat),
        .n2b_flat(n2b_flat),
        .fc1_w_flat(fc1_w_flat),
        .fc1_b_flat(fc1_b_flat),
        .fc2_w_flat(fc2_w_flat),
        .fc2_b_flat(fc2_b_flat),
        .norm2_flat(norm2_flat),
        .fc1_flat(fc1_flat),
        .gelu_flat(gelu_flat),
        .fc2_flat(fc2_flat),
        .y_flat(y_flat)
    );

endmodule
