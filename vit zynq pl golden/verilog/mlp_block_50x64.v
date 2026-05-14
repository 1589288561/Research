`include "fixed_params.vh"

module mlp_block_50x64 (
    input  signed [`MAT5X4_W-1:0] x_flat,
    input  signed [`VEC4_W-1:0]   n2w_flat,
    input  signed [`VEC4_W-1:0]   n2b_flat,
    input  signed [`W8X4_W-1:0]   fc1_w_flat,
    input  signed [`VEC8_W-1:0]   fc1_b_flat,
    input  signed [`W4X8_W-1:0]   fc2_w_flat,
    input  signed [`VEC4_W-1:0]   fc2_b_flat,

    output signed [`MAT5X4_W-1:0] norm2_flat,
    output signed [`MAT5X8_W-1:0] fc1_flat,
    output signed [`MAT5X8_W-1:0] gelu_flat,
    output signed [`MAT5X4_W-1:0] fc2_flat,
    output signed [`MAT5X4_W-1:0] y_flat
);

    norm_stage_50x64 u_norm2 (
        .x_flat(x_flat),
        .w_flat(n2w_flat),
        .b_flat(n2b_flat),
        .y_flat(norm2_flat)
    );

    fc1_stage_50x64_to_50x128 u_fc1 (
        .x_flat(norm2_flat),
        .w_flat(fc1_w_flat),
        .b_flat(fc1_b_flat),
        .y_flat(fc1_flat)
    );

    gelu_stage_50x128 u_gelu (
        .x_flat(fc1_flat),
        .y_flat(gelu_flat)
    );

    fc2_stage_50x128_to_50x64 u_fc2 (
        .x_flat(gelu_flat),
        .w_flat(fc2_w_flat),
        .b_flat(fc2_b_flat),
        .y_flat(fc2_flat)
    );

    residual_add_50x64 u_res2 (
        .a_flat(x_flat),
        .b_flat(fc2_flat),
        .y_flat(y_flat)
    );

endmodule
