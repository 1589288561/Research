`include "fixed_params.vh"

module classifier_head_64x10 (
    input  signed [`MAT_T_D_W-1:0] pre_classifier_flat,

    input  signed [`W_D_D_W-1:0]   fc1_w_flat,
    input  signed [`VEC_D_W-1:0]   fc1_b_flat,
    input  signed [`W_C_D_W-1:0]   fc2_w_flat,
    input  signed [`VEC_C_W-1:0]   fc2_b_flat,

    output reg signed [`VEC_D_W-1:0] cls_input_flat,
    output reg signed [`VEC_D_W-1:0] fc1_flat,
    output reg signed [`VEC_D_W-1:0] tanh_flat,
    output reg signed [`VEC_C_W-1:0] logits_flat,
    output reg [3:0] pred_class
);

    `include "tanh_lut_q412_full.vh"

    reg signed [`WIDTH-1:0] cls_input [0:`D-1];
    reg signed [`WIDTH-1:0] fc1_w [0:`D-1][0:`D-1];
    reg signed [`WIDTH-1:0] fc2_w [0:`N_CLASSES-1][0:`D-1];
    reg signed [`WIDTH-1:0] fc1_b [0:`D-1];
    reg signed [`WIDTH-1:0] fc2_b [0:`N_CLASSES-1];
    reg signed [`WIDTH-1:0] fc1_y [0:`D-1];
    reg signed [`WIDTH-1:0] tanh_y [0:`D-1];
    reg signed [`WIDTH-1:0] logits [0:`N_CLASSES-1];

    integer i, j, c;
    reg signed [63:0] acc;
    reg signed [`WIDTH-1:0] best_logit;

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
        cls_input_flat = {`VEC_D_W{1'b0}};
        fc1_flat = {`VEC_D_W{1'b0}};
        tanh_flat = {`VEC_D_W{1'b0}};
        logits_flat = {`VEC_C_W{1'b0}};
        pred_class = 4'd0;

        for (i = 0; i < `D; i = i + 1) begin
            cls_input[i] = pre_classifier_flat[(i+1)*`WIDTH-1 -: `WIDTH];
            fc1_b[i] = fc1_b_flat[(i+1)*`WIDTH-1 -: `WIDTH];
            for (j = 0; j < `D; j = j + 1)
                fc1_w[i][j] = fc1_w_flat[((i*`D+j)+1)*`WIDTH-1 -: `WIDTH];
        end

        for (c = 0; c < `N_CLASSES; c = c + 1) begin
            fc2_b[c] = fc2_b_flat[(c+1)*`WIDTH-1 -: `WIDTH];
            for (j = 0; j < `D; j = j + 1)
                fc2_w[c][j] = fc2_w_flat[((c*`D+j)+1)*`WIDTH-1 -: `WIDTH];
        end

        for (i = 0; i < `D; i = i + 1) begin
            acc = 0;
            for (j = 0; j < `D; j = j + 1)
                acc = acc + cls_input[j] * fc1_w[i][j];
            acc = acc + ({{48{fc1_b[i][`WIDTH-1]}}, fc1_b[i]} <<< `FRAC);
            fc1_y[i] = sat16_from64(acc >>> `FRAC);
            tanh_y[i] = tanh_lut_lookup(fc1_y[i]);
        end

        for (c = 0; c < `N_CLASSES; c = c + 1) begin
            acc = 0;
            for (j = 0; j < `D; j = j + 1)
                acc = acc + tanh_y[j] * fc2_w[c][j];
            acc = acc + ({{48{fc2_b[c][`WIDTH-1]}}, fc2_b[c]} <<< `FRAC);
            logits[c] = sat16_from64(acc >>> `FRAC);
        end

        best_logit = logits[0];
        for (c = 1; c < `N_CLASSES; c = c + 1) begin
            if (logits[c] > best_logit) begin
                best_logit = logits[c];
                pred_class = c[3:0];
            end
        end

        for (i = 0; i < `D; i = i + 1) begin
            cls_input_flat[(i+1)*`WIDTH-1 -: `WIDTH] = cls_input[i];
            fc1_flat[(i+1)*`WIDTH-1 -: `WIDTH] = fc1_y[i];
            tanh_flat[(i+1)*`WIDTH-1 -: `WIDTH] = tanh_y[i];
        end

        for (c = 0; c < `N_CLASSES; c = c + 1)
            logits_flat[(c+1)*`WIDTH-1 -: `WIDTH] = logits[c];
    end

endmodule
