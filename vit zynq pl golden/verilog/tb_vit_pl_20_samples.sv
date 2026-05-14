`timescale 1ns/1ps
`include "fixed_params.vh"

module tb_vit_pl_20_samples_common #(
    parameter string WEIGHT_ROOT = "../exports/seed1234",
    parameter string DATA_ROOT   = "../pl_io_20_samples/seed1234/fixed"
);

    `include "tb_encoder_io.svh"

    reg signed [`MAT_T_D_W-1:0] x_flat;

    reg signed [`VEC_D_W-1:0]   n1w_flat     [0:5];
    reg signed [`VEC_D_W-1:0]   n1b_flat     [0:5];
    reg signed [`W_D_D_W-1:0]   wq_flat      [0:5];
    reg signed [`VEC_D_W-1:0]   bq_flat      [0:5];
    reg signed [`W_D_D_W-1:0]   wk_flat      [0:5];
    reg signed [`VEC_D_W-1:0]   bk_flat      [0:5];
    reg signed [`W_D_D_W-1:0]   wv_flat      [0:5];
    reg signed [`VEC_D_W-1:0]   bv_flat      [0:5];

    reg signed [`VEC_D_W-1:0]   n2w_flat     [0:5];
    reg signed [`VEC_D_W-1:0]   n2b_flat     [0:5];
    reg signed [`W_DFF_D_W-1:0] fc1_w_flat   [0:5];
    reg signed [`VEC_DFF_W-1:0] fc1_b_flat   [0:5];
    reg signed [`W_D_DFF_W-1:0] fc2_w_flat   [0:5];
    reg signed [`VEC_D_W-1:0]   fc2_b_flat   [0:5];

    reg signed [`VEC_D_W-1:0] encoder_norm_w_flat;
    reg signed [`VEC_D_W-1:0] encoder_norm_b_flat;
    reg signed [`VEC_D_W-1:0] model_norm_w_flat;
    reg signed [`VEC_D_W-1:0] model_norm_b_flat;

    reg signed [`W_D_D_W-1:0] classifier_fc1_w_flat;
    reg signed [`VEC_D_W-1:0] classifier_fc1_b_flat;
    reg signed [`W_C_D_W-1:0] classifier_fc2_w_flat;
    reg signed [`VEC_C_W-1:0] classifier_fc2_b_flat;

    reg signed [`VEC_C_W-1:0] logits_ref_flat;
    integer pred_ref;
    integer label_ref;

    wire signed [`MAT_T_D_W-1:0] block_out_flat [0:5];
    wire signed [`MAT_T_D_W-1:0] encoder_norm_flat;
    wire signed [`MAT_T_D_W-1:0] pre_classifier_flat;
    wire signed [`VEC_D_W-1:0] cls_input_flat;
    wire signed [`VEC_D_W-1:0] classifier_fc1_flat;
    wire signed [`VEC_D_W-1:0] classifier_tanh_flat;
    wire signed [`VEC_C_W-1:0] logits_flat;
    wire [3:0] pred_class;

    integer blk;
    integer sample_idx;
    integer sample_count;
    integer mismatch_count;
    integer logits_diff;
    integer overall_logits_max_diff;
    integer fd_embedding;
    integer fd_logits;
    integer fd_pred;
    integer fd_label;
    string base;

    vit_pl_golden_top dut (
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

        .classifier_fc1_w_flat(classifier_fc1_w_flat),
        .classifier_fc1_b_flat(classifier_fc1_b_flat),
        .classifier_fc2_w_flat(classifier_fc2_w_flat),
        .classifier_fc2_b_flat(classifier_fc2_b_flat),

        .block_out_flat(block_out_flat),
        .encoder_norm_flat(encoder_norm_flat),
        .pre_classifier_flat(pre_classifier_flat),
        .cls_input_flat(cls_input_flat),
        .classifier_fc1_flat(classifier_fc1_flat),
        .classifier_tanh_flat(classifier_tanh_flat),
        .logits_flat(logits_flat),
        .pred_class(pred_class)
    );

    task automatic load_vec_c(input string path, output reg signed [`VEC_C_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = 0;
            fd = $fopen(path, "r");
            if (fd == 0) begin
                $display("ERROR: cannot open %s", path);
                $finish;
            end
            for (idx = 0; idx < `N_CLASSES; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) begin
                    $display("ERROR: reading %s idx=%0d", path, idx);
                    $finish;
                end
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    task automatic load_w_c_d(input string path, output reg signed [`W_C_D_W-1:0] flat);
        integer fd, r, val, idx;
        begin
            flat = 0;
            fd = $fopen(path, "r");
            if (fd == 0) begin
                $display("ERROR: cannot open %s", path);
                $finish;
            end
            for (idx = 0; idx < `N_CLASSES * `D; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) begin
                    $display("ERROR: reading %s idx=%0d", path, idx);
                    $finish;
                end
                `ELEM(flat, idx) = val;
            end
            $fclose(fd);
        end
    endtask

    task automatic load_int_scalar(input string path, output integer val);
        integer fd, r;
        begin
            fd = $fopen(path, "r");
            if (fd == 0) begin
                $display("ERROR: cannot open %s", path);
                $finish;
            end
            r = $fscanf(fd, "%d", val);
            if (r != 1) begin
                $display("ERROR: reading %s", path);
                $finish;
            end
            $fclose(fd);
        end
    endtask

    task automatic read_mat_t_d_from_fd(input integer fd, output reg signed [`MAT_T_D_W-1:0] flat);
        integer r, val, idx;
        begin
            flat = 0;
            for (idx = 0; idx < `T * `D; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) begin
                    $display("ERROR: reading embedding sample idx=%0d", sample_idx);
                    $finish;
                end
                `ELEM(flat, idx) = val;
            end
        end
    endtask

    task automatic read_vec_c_from_fd(input integer fd, output reg signed [`VEC_C_W-1:0] flat);
        integer r, val, idx;
        begin
            flat = 0;
            for (idx = 0; idx < `N_CLASSES; idx = idx + 1) begin
                r = $fscanf(fd, "%d", val);
                if (r != 1) begin
                    $display("ERROR: reading logits sample idx=%0d", sample_idx);
                    $finish;
                end
                `ELEM(flat, idx) = val;
            end
        end
    endtask

    task automatic read_int_from_fd(input integer fd, output integer val);
        integer r;
        begin
            r = $fscanf(fd, "%d", val);
            if (r != 1) begin
                $display("ERROR: reading integer sample idx=%0d", sample_idx);
                $finish;
            end
        end
    endtask

    task automatic calc_logits_diff(
        input signed [`VEC_C_W-1:0] actual,
        input signed [`VEC_C_W-1:0] expected,
        output integer max_abs_diff
    );
        integer i, diff;
        begin
            max_abs_diff = 0;
            for (i = 0; i < `N_CLASSES; i = i + 1) begin
                diff = $signed(`ELEM(actual, i)) - $signed(`ELEM(expected, i));
                if (diff < 0) diff = -diff;
                if (diff > max_abs_diff) max_abs_diff = diff;
            end
        end
    endtask

    task automatic load_weights;
        begin
            for (blk = 0; blk < 6; blk = blk + 1) begin
                base = $sformatf("%s/block%0d/basic/fixed", WEIGHT_ROOT, blk + 1);

                load_vec_d({base, "/norm1_weight.txt"}, n1w_flat[blk]);
                load_vec_d({base, "/norm1_bias.txt"}, n1b_flat[blk]);
                load_w_d_d({base, "/wq_weight.txt"}, wq_flat[blk]);
                load_vec_d({base, "/wq_bias.txt"}, bq_flat[blk]);
                load_w_d_d({base, "/wk_weight.txt"}, wk_flat[blk]);
                load_vec_d({base, "/wk_bias.txt"}, bk_flat[blk]);
                load_w_d_d({base, "/wv_weight.txt"}, wv_flat[blk]);
                load_vec_d({base, "/wv_bias.txt"}, bv_flat[blk]);

                load_vec_d({base, "/norm2_weight.txt"}, n2w_flat[blk]);
                load_vec_d({base, "/norm2_bias.txt"}, n2b_flat[blk]);
                load_w_dff_d({base, "/fc1_weight.txt"}, fc1_w_flat[blk]);
                load_vec_dff({base, "/fc1_bias.txt"}, fc1_b_flat[blk]);
                load_w_d_dff({base, "/fc2_weight.txt"}, fc2_w_flat[blk]);
                load_vec_d({base, "/fc2_bias.txt"}, fc2_b_flat[blk]);
            end

            load_vec_d({WEIGHT_ROOT, "/encoder_final_norm/fixed/norm_weight.txt"}, encoder_norm_w_flat);
            load_vec_d({WEIGHT_ROOT, "/encoder_final_norm/fixed/norm_bias.txt"}, encoder_norm_b_flat);
            load_vec_d({WEIGHT_ROOT, "/model_norm/fixed/norm_weight.txt"}, model_norm_w_flat);
            load_vec_d({WEIGHT_ROOT, "/model_norm/fixed/norm_bias.txt"}, model_norm_b_flat);

            load_w_d_d({WEIGHT_ROOT, "/classifier/fixed/classifier_fc1_weight.txt"}, classifier_fc1_w_flat);
            load_vec_d({WEIGHT_ROOT, "/classifier/fixed/classifier_fc1_bias.txt"}, classifier_fc1_b_flat);
            load_w_c_d({WEIGHT_ROOT, "/classifier/fixed/classifier_fc2_weight.txt"}, classifier_fc2_w_flat);
            load_vec_c({WEIGHT_ROOT, "/classifier/fixed/classifier_fc2_bias.txt"}, classifier_fc2_b_flat);
        end
    endtask

    initial begin
        load_weights();
        load_int_scalar({DATA_ROOT, "/sample_count.txt"}, sample_count);

        fd_embedding = $fopen({DATA_ROOT, "/embedding_output_20.txt"}, "r");
        fd_logits = $fopen({DATA_ROOT, "/logits_20.txt"}, "r");
        fd_pred = $fopen({DATA_ROOT, "/pred_class_20.txt"}, "r");
        fd_label = $fopen({DATA_ROOT, "/label_20.txt"}, "r");

        if (fd_embedding == 0 || fd_logits == 0 || fd_pred == 0 || fd_label == 0) begin
            $display("ERROR: cannot open 20-sample PL IO files under %s", DATA_ROOT);
            $finish;
        end

        mismatch_count = 0;
        overall_logits_max_diff = 0;

        $display("");
        $display("================ 20-SAMPLE PL CHECK ================");
        $display("WEIGHT_ROOT=%s", WEIGHT_ROOT);
        $display("DATA_ROOT=%s", DATA_ROOT);
        $display("sample_count=%0d", sample_count);

        for (sample_idx = 0; sample_idx < sample_count; sample_idx = sample_idx + 1) begin
            read_mat_t_d_from_fd(fd_embedding, x_flat);
            read_vec_c_from_fd(fd_logits, logits_ref_flat);
            read_int_from_fd(fd_pred, pred_ref);
            read_int_from_fd(fd_label, label_ref);

            #20;

            calc_logits_diff(logits_flat, logits_ref_flat, logits_diff);
            if (logits_diff > overall_logits_max_diff)
                overall_logits_max_diff = logits_diff;

            if (pred_class !== pred_ref[3:0]) begin
                mismatch_count = mismatch_count + 1;
                $display(
                    "MISMATCH sample=%0d label=%0d expected_pred=%0d actual_pred=%0d logits_max_diff=%0d",
                    sample_idx, label_ref, pred_ref, pred_class, logits_diff
                );
            end else begin
                $display(
                    "PASS sample=%0d label=%0d pred=%0d logits_max_diff=%0d",
                    sample_idx, label_ref, pred_class, logits_diff
                );
            end
        end

        $fclose(fd_embedding);
        $fclose(fd_logits);
        $fclose(fd_pred);
        $fclose(fd_label);

        $display("");
        $display("================ 20-SAMPLE SUMMARY ================");
        $display("checked_samples=%0d", sample_count);
        $display("pred_mismatch_count=%0d", mismatch_count);
        $display("overall_logits_max_abs_diff=%0d", overall_logits_max_diff);

        if (mismatch_count == 0)
            $display("[20_samples] PASS: all predicted classes match reference.");
        else
            $display("[20_samples] FAIL: predicted class mismatches detected.");

        $finish;
    end

endmodule

module tb_vit_pl_20_samples_seed1234;
    tb_vit_pl_20_samples_common #(
        .WEIGHT_ROOT("../exports/seed1234"),
        .DATA_ROOT("../pl_io_20_samples/seed1234/fixed")
    ) run();
endmodule

module tb_vit_pl_20_samples_seed2026;
    tb_vit_pl_20_samples_common #(
        .WEIGHT_ROOT("../exports/seed2026"),
        .DATA_ROOT("../pl_io_20_samples/seed2026/fixed")
    ) run();
endmodule
