`include "fixed_params.vh"

module att_core_50x64 (
    input  signed [`MAT_T_D_W-1:0] x_flat,

    input  signed [`W_D_D_W-1:0]   wq_flat,
    input  signed [`VEC_D_W-1:0]   bq_flat,

    input  signed [`W_D_D_W-1:0]   wk_flat,
    input  signed [`VEC_D_W-1:0]   bk_flat,

    input  signed [`W_D_D_W-1:0]   wv_flat,
    input  signed [`VEC_D_W-1:0]   bv_flat,

    output reg signed [`MAT_T_D_W-1:0] ctx_flat
);

    `include "exp_lut_q412_full.vh"

    reg signed [`WIDTH-1:0] x  [0:`T-1][0:`D-1];
    reg signed [`WIDTH-1:0] wq [0:`D-1][0:`D-1];
    reg signed [`WIDTH-1:0] wk [0:`D-1][0:`D-1];
    reg signed [`WIDTH-1:0] wv [0:`D-1][0:`D-1];
    reg signed [`WIDTH-1:0] bq [0:`D-1];
    reg signed [`WIDTH-1:0] bk [0:`D-1];
    reg signed [`WIDTH-1:0] bv [0:`D-1];

    reg signed [`WIDTH-1:0] q   [0:`T-1][0:`D-1];
    reg signed [`WIDTH-1:0] k   [0:`T-1][0:`D-1];
    reg signed [`WIDTH-1:0] v   [0:`T-1][0:`D-1];
    reg signed [`WIDTH-1:0] ctx [0:`T-1][0:`D-1];

    // Linear outputs before dropping from Q8.24 to Q4.12. Scores use these
    // higher precision values, matching the small-scale model's intent.
    reg signed [63:0] q_full [0:`T-1][0:`D-1];
    reg signed [63:0] k_full [0:`T-1][0:`D-1];

    reg signed [`WIDTH-1:0] score [0:`HEADS-1][0:`T-1][0:`T-1];
    reg signed [`WIDTH-1:0] prob  [0:`HEADS-1][0:`T-1][0:`T-1];
    reg signed [`WIDTH-1:0] exp_val [0:`T-1];

    integer h, i, j, d, t;
    integer base_d;
    reg signed [63:0] acc;
    reg signed [63:0] score_acc;
    reg signed [`WIDTH-1:0] maxv;
    reg signed [`WIDTH-1:0] z;
    reg [31:0] sum_exp;

    function automatic signed [`WIDTH-1:0] sat16_from64;
        input signed [63:0] x_in;
        begin
            if (x_in > 64'sd32767)
                sat16_from64 = 16'sd32767;
            else if (x_in < -64'sd32768)
                sat16_from64 = -16'sd32768;
            else
                sat16_from64 = x_in[`WIDTH-1:0];
        end
    endfunction

    always @(*) begin
        ctx_flat = {`MAT_T_D_W{1'b0}};

        for (i = 0; i < `T; i = i + 1)
            for (j = 0; j < `D; j = j + 1)
                x[i][j] = x_flat[((i*`D+j)+1)*`WIDTH-1 -: `WIDTH];

        for (i = 0; i < `D; i = i + 1) begin
            for (j = 0; j < `D; j = j + 1) begin
                wq[i][j] = wq_flat[((i*`D+j)+1)*`WIDTH-1 -: `WIDTH];
                wk[i][j] = wk_flat[((i*`D+j)+1)*`WIDTH-1 -: `WIDTH];
                wv[i][j] = wv_flat[((i*`D+j)+1)*`WIDTH-1 -: `WIDTH];
            end
            bq[i] = bq_flat[(i+1)*`WIDTH-1 -: `WIDTH];
            bk[i] = bk_flat[(i+1)*`WIDTH-1 -: `WIDTH];
            bv[i] = bv_flat[(i+1)*`WIDTH-1 -: `WIDTH];
        end

        // Q / K / V = X * W^T + b, with weights stored out-feature first.
        for (i = 0; i < `T; i = i + 1) begin
            for (j = 0; j < `D; j = j + 1) begin
                acc = 0;
                for (d = 0; d < `D; d = d + 1)
                    acc = acc + x[i][d] * wq[j][d];
                acc = acc + ({{48{bq[j][`WIDTH-1]}}, bq[j]} <<< `FRAC);
                q_full[i][j] = acc;
                q[i][j] = sat16_from64(acc >>> `FRAC);

                acc = 0;
                for (d = 0; d < `D; d = d + 1)
                    acc = acc + x[i][d] * wk[j][d];
                acc = acc + ({{48{bk[j][`WIDTH-1]}}, bk[j]} <<< `FRAC);
                k_full[i][j] = acc;
                k[i][j] = sat16_from64(acc >>> `FRAC);

                acc = 0;
                for (d = 0; d < `D; d = d + 1)
                    acc = acc + x[i][d] * wv[j][d];
                acc = acc + ({{48{bv[j][`WIDTH-1]}}, bv[j]} <<< `FRAC);
                v[i][j] = sat16_from64(acc >>> `FRAC);
            end
        end

        // PyTorch shape path:
        // [T,D] -> view [T,HEADS,HEAD_DIM] -> transpose -> [HEADS,T,HEAD_DIM].
        for (h = 0; h < `HEADS; h = h + 1) begin
            base_d = h * `HEAD_DIM;

            for (i = 0; i < `T; i = i + 1) begin
                for (j = 0; j < `T; j = j + 1) begin
                    score_acc = 0;
                    for (d = 0; d < `HEAD_DIM; d = d + 1)
                        score_acc = score_acc + q_full[i][base_d+d] * k_full[j][base_d+d];

                    // q_full/k_full are Q8.24. Product sum is Q16.48.
                    // >>>36 returns Q4.12, then /sqrt(HEAD_DIM=16) is >>>2.
                    score[h][i][j] = sat16_from64((score_acc >>> 36) >>> 2);
                end
            end

            for (i = 0; i < `T; i = i + 1) begin
                maxv = score[h][i][0];
                for (j = 1; j < `T; j = j + 1)
                    if (score[h][i][j] > maxv)
                        maxv = score[h][i][j];

                sum_exp = 0;
                for (j = 0; j < `T; j = j + 1) begin
                    z = score[h][i][j] - maxv;
                    exp_val[j] = exp_lut_lookup(z);
                    sum_exp = sum_exp + exp_val[j];
                end

                for (j = 0; j < `T; j = j + 1)
                    prob[h][i][j] = (exp_val[j] <<< `FRAC) / sum_exp;
            end

            for (i = 0; i < `T; i = i + 1) begin
                for (d = 0; d < `HEAD_DIM; d = d + 1) begin
                    acc = 0;
                    for (t = 0; t < `T; t = t + 1)
                        acc = acc + prob[h][i][t] * v[t][base_d+d];
                    ctx[i][base_d+d] = sat16_from64(acc >>> `FRAC);
                end
            end
        end

        for (i = 0; i < `T; i = i + 1)
            for (j = 0; j < `D; j = j + 1)
                ctx_flat[((i*`D+j)+1)*`WIDTH-1 -: `WIDTH] = ctx[i][j];
    end

endmodule
