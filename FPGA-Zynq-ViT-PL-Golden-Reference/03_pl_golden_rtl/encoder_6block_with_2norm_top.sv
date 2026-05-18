`include "fixed_params.vh"

module encoder_6block_with_2norm_top (
    input signed [`MAT_T_D_W-1:0] x_flat,

    input signed [`VEC_D_W-1:0]   n1w_flat     [0:5],
    input signed [`VEC_D_W-1:0]   n1b_flat     [0:5],
    input signed [`W_D_D_W-1:0]   wq_flat      [0:5],
    input signed [`VEC_D_W-1:0]   bq_flat      [0:5],
    input signed [`W_D_D_W-1:0]   wk_flat      [0:5],
    input signed [`VEC_D_W-1:0]   bk_flat      [0:5],
    input signed [`W_D_D_W-1:0]   wv_flat      [0:5],
    input signed [`VEC_D_W-1:0]   bv_flat      [0:5],

    input signed [`VEC_D_W-1:0]   n2w_flat     [0:5],
    input signed [`VEC_D_W-1:0]   n2b_flat     [0:5],
    input signed [`W_DFF_D_W-1:0] fc1_w_flat   [0:5],
    input signed [`VEC_DFF_W-1:0] fc1_b_flat   [0:5],
    input signed [`W_D_DFF_W-1:0] fc2_w_flat   [0:5],
    input signed [`VEC_D_W-1:0]   fc2_b_flat   [0:5],

    input signed [`VEC_D_W-1:0]   encoder_norm_w_flat,
    input signed [`VEC_D_W-1:0]   encoder_norm_b_flat,
    input signed [`VEC_D_W-1:0]   model_norm_w_flat,
    input signed [`VEC_D_W-1:0]   model_norm_b_flat,

    output signed [`MAT_T_D_W-1:0] block_out_flat [0:5],
    output signed [`MAT_T_D_W-1:0] encoder_norm_flat,
    output signed [`MAT_T_D_W-1:0] y_flat
);

    wire signed [`MAT_T_D_W-1:0] block_in_flat [0:6];
    wire signed [`MAT_T_D_W-1:0] norm1_flat    [0:5];
    wire signed [`MAT_T_D_W-1:0] att_only_flat [0:5];
    wire signed [`MAT_T_D_W-1:0] att_out_flat  [0:5];
    wire signed [`MAT_T_D_W-1:0] norm2_flat    [0:5];
    wire signed [`MAT_T_DFF_W-1:0] fc1_flat    [0:5];
    wire signed [`MAT_T_DFF_W-1:0] gelu_flat   [0:5];
    wire signed [`MAT_T_D_W-1:0] fc2_flat      [0:5];

    assign block_in_flat[0] = x_flat;

    genvar i;
    generate
        for (i = 0; i < 6; i = i + 1) begin : gen_encoder_blocks
            encoder_block_top u_block (
                .x_flat(block_in_flat[i]),

                .n1w_flat(n1w_flat[i]),
                .n1b_flat(n1b_flat[i]),
                .wq_flat(wq_flat[i]),
                .bq_flat(bq_flat[i]),
                .wk_flat(wk_flat[i]),
                .bk_flat(bk_flat[i]),
                .wv_flat(wv_flat[i]),
                .bv_flat(bv_flat[i]),

                .n2w_flat(n2w_flat[i]),
                .n2b_flat(n2b_flat[i]),
                .fc1_w_flat(fc1_w_flat[i]),
                .fc1_b_flat(fc1_b_flat[i]),
                .fc2_w_flat(fc2_w_flat[i]),
                .fc2_b_flat(fc2_b_flat[i]),

                .norm1_flat(norm1_flat[i]),
                .att_only_flat(att_only_flat[i]),
                .att_out_flat(att_out_flat[i]),
                .norm2_flat(norm2_flat[i]),
                .fc1_flat(fc1_flat[i]),
                .gelu_flat(gelu_flat[i]),
                .fc2_flat(fc2_flat[i]),
                .y_flat(block_out_flat[i])
            );

            assign block_in_flat[i + 1] = block_out_flat[i];
        end
    endgenerate

    norm_stage_50x64 u_encoder_final_norm (
        .x_flat(block_in_flat[6]),
        .w_flat(encoder_norm_w_flat),
        .b_flat(encoder_norm_b_flat),
        .y_flat(encoder_norm_flat)
    );

    norm_stage_50x64 u_model_norm (
        .x_flat(encoder_norm_flat),
        .w_flat(model_norm_w_flat),
        .b_flat(model_norm_b_flat),
        .y_flat(y_flat)
    );

endmodule

