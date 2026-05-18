`include "fixed_params.vh"

module vit_pl_golden_top (
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

    input signed [`W_D_D_W-1:0]   classifier_fc1_w_flat,
    input signed [`VEC_D_W-1:0]   classifier_fc1_b_flat,
    input signed [`W_C_D_W-1:0]   classifier_fc2_w_flat,
    input signed [`VEC_C_W-1:0]   classifier_fc2_b_flat,

    output signed [`MAT_T_D_W-1:0] block_out_flat [0:5],
    output signed [`MAT_T_D_W-1:0] encoder_norm_flat,
    output signed [`MAT_T_D_W-1:0] pre_classifier_flat,
    output signed [`VEC_D_W-1:0]   cls_input_flat,
    output signed [`VEC_D_W-1:0]   classifier_fc1_flat,
    output signed [`VEC_D_W-1:0]   classifier_tanh_flat,
    output signed [`VEC_C_W-1:0]   logits_flat,
    output [3:0]                   pred_class
);

    encoder_6block_with_2norm_top u_encoder (
        .x_flat(x_flat),

        .n1w_flat(n1w_flat),
        .n1b_flat(n1b_flat),
        .wq_flat(wq_flat),
        .bq_flat(bq_flat),
        .wk_flat(wk_flat),
        .bk_flat(bk_flat),
        .wv_flat(wv_flat),
        .bv_flat(bv_flat),

        .n2w_flat(n2w_flat),
        .n2b_flat(n2b_flat),
        .fc1_w_flat(fc1_w_flat),
        .fc1_b_flat(fc1_b_flat),
        .fc2_w_flat(fc2_w_flat),
        .fc2_b_flat(fc2_b_flat),

        .encoder_norm_w_flat(encoder_norm_w_flat),
        .encoder_norm_b_flat(encoder_norm_b_flat),
        .model_norm_w_flat(model_norm_w_flat),
        .model_norm_b_flat(model_norm_b_flat),

        .block_out_flat(block_out_flat),
        .encoder_norm_flat(encoder_norm_flat),
        .y_flat(pre_classifier_flat)
    );

    classifier_head_64x10 u_classifier (
        .pre_classifier_flat(pre_classifier_flat),
        .fc1_w_flat(classifier_fc1_w_flat),
        .fc1_b_flat(classifier_fc1_b_flat),
        .fc2_w_flat(classifier_fc2_w_flat),
        .fc2_b_flat(classifier_fc2_b_flat),
        .cls_input_flat(cls_input_flat),
        .fc1_flat(classifier_fc1_flat),
        .tanh_flat(classifier_tanh_flat),
        .logits_flat(logits_flat),
        .pred_class(pred_class)
    );

endmodule
