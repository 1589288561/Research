`timescale 1ns/1ps
`include "fixed_params.vh"

module tb_vit_pl_golden_basic_common #(
    parameter string ROOT = "../exports/seed1234"
);

    `include "tb_encoder_io.svh"

    localparam integer SAMPLE_N = 20;

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

    reg signed [`MAT_T_D_W-1:0] block_ref_flat [0:5];
    reg signed [`MAT_T_D_W-1:0] pre_classifier_ref_flat;
    reg signed [`VEC_D_W-1:0] cls_input_ref_flat;
    reg signed [`VEC_D_W-1:0] classifier_fc1_ref_flat;
    reg signed [`VEC_D_W-1:0] classifier_tanh_ref_flat;
    reg signed [`VEC_C_W-1:0] logits_ref_flat;
    integer pred_ref;

    wire signed [`MAT_T_D_W-1:0] block_out_flat [0:5];
    wire signed [`MAT_T_D_W-1:0] encoder_norm_flat;
    wire signed [`MAT_T_D_W-1:0] pre_classifier_flat;
    wire signed [`VEC_D_W-1:0] cls_input_flat;
    wire signed [`VEC_D_W-1:0] classifier_fc1_flat;
    wire signed [`VEC_D_W-1:0] classifier_tanh_flat;
    wire signed [`VEC_C_W-1:0] logits_flat;
    wire [3:0] pred_class;

    integer blk;
    integer max_diff;
    integer overall_max_diff;
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

    task automatic report_first_n_vec_d(
        input string name,
        input signed [`VEC_D_W-1:0] actual,
        input signed [`VEC_D_W-1:0] expected,
        input integer n_items,
        output integer max_abs_diff
    );
        integer i, expected_val, actual_val, diff, abs_diff;
        begin
            max_abs_diff = 0;
            $display("======== %s first %0d values ========", name, n_items);
            $display("  idx   expected   actual   diff");
            for (i = 0; i < n_items; i = i + 1) begin
                expected_val = $signed(`ELEM(expected, i));
                actual_val = $signed(`ELEM(actual, i));
                diff = actual_val - expected_val;
                abs_diff = diff;
                if (abs_diff < 0) abs_diff = -abs_diff;
                if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
                $display("%5d %10d %8d %7d", i, expected_val, actual_val, diff);
            end
            $display("[%s] first_%0d_max_abs_diff=%0d", name, n_items, max_abs_diff);
        end
    endtask

    task automatic report_vec_c(
        input string name,
        input signed [`VEC_C_W-1:0] actual,
        input signed [`VEC_C_W-1:0] expected,
        output integer max_abs_diff
    );
        integer i, expected_val, actual_val, diff, abs_diff;
        begin
            max_abs_diff = 0;
            $display("======== %s logits ========", name);
            $display("class   expected   actual   diff");
            for (i = 0; i < `N_CLASSES; i = i + 1) begin
                expected_val = $signed(`ELEM(expected, i));
                actual_val = $signed(`ELEM(actual, i));
                diff = actual_val - expected_val;
                abs_diff = diff;
                if (abs_diff < 0) abs_diff = -abs_diff;
                if (abs_diff > max_abs_diff) max_abs_diff = abs_diff;
                $display("%5d %10d %8d %7d", i, expected_val, actual_val, diff);
            end
            $display("[%s] max_abs_diff=%0d", name, max_abs_diff);
        end
    endtask

    task automatic update_overall(input integer diff);
        begin
            if (diff > overall_max_diff) overall_max_diff = diff;
        end
    endtask

    initial begin
        overall_max_diff = 0;

        load_mat_t_d({ROOT, "/embedding_output/fixed/embedding_output.txt"}, x_flat);

        for (blk = 0; blk < 6; blk = blk + 1) begin
            base = $sformatf("%s/block%0d/basic/fixed", ROOT, blk + 1);

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

            load_mat_t_d({base, "/block_output.txt"}, block_ref_flat[blk]);
        end

        load_vec_d({ROOT, "/encoder_final_norm/fixed/norm_weight.txt"}, encoder_norm_w_flat);
        load_vec_d({ROOT, "/encoder_final_norm/fixed/norm_bias.txt"}, encoder_norm_b_flat);
        load_vec_d({ROOT, "/model_norm/fixed/norm_weight.txt"}, model_norm_w_flat);
        load_vec_d({ROOT, "/model_norm/fixed/norm_bias.txt"}, model_norm_b_flat);
        load_mat_t_d({ROOT, "/pre_classifier/fixed/pre_classifier.txt"}, pre_classifier_ref_flat);

        load_w_d_d({ROOT, "/classifier/fixed/classifier_fc1_weight.txt"}, classifier_fc1_w_flat);
        load_vec_d({ROOT, "/classifier/fixed/classifier_fc1_bias.txt"}, classifier_fc1_b_flat);
        load_w_c_d({ROOT, "/classifier/fixed/classifier_fc2_weight.txt"}, classifier_fc2_w_flat);
        load_vec_c({ROOT, "/classifier/fixed/classifier_fc2_bias.txt"}, classifier_fc2_b_flat);
        load_vec_d({ROOT, "/classifier/fixed/cls_input.txt"}, cls_input_ref_flat);
        load_vec_d({ROOT, "/classifier/fixed/classifier_fc1_output.txt"}, classifier_fc1_ref_flat);
        load_vec_d({ROOT, "/classifier/fixed/classifier_tanh_output.txt"}, classifier_tanh_ref_flat);
        load_vec_c({ROOT, "/classifier/fixed/logits.txt"}, logits_ref_flat);
        load_int_scalar({ROOT, "/classifier/fixed/pred_class.txt"}, pred_ref);

        #20;

        $display("");
        $display("================ PL FINAL OUTPUT CHECK ================");
        report_vec_c("logits", logits_flat, logits_ref_flat, max_diff);
        update_overall(max_diff);
        $display("[pred_class] expected=%0d actual=%0d diff=%0d", pred_ref, pred_class, pred_class - pred_ref);

        $display("");
        $display("================ PL BOUNDARY DEBUG CHECK ================");
        report_first_n_mat_t_d("pre_classifier", pre_classifier_flat, pre_classifier_ref_flat, SAMPLE_N, max_diff);
        update_overall(max_diff);
        report_first_n_vec_d("cls_input", cls_input_flat, cls_input_ref_flat, SAMPLE_N, max_diff);
        update_overall(max_diff);
        report_first_n_vec_d("classifier_fc1", classifier_fc1_flat, classifier_fc1_ref_flat, SAMPLE_N, max_diff);
        update_overall(max_diff);
        report_first_n_vec_d("classifier_tanh", classifier_tanh_flat, classifier_tanh_ref_flat, SAMPLE_N, max_diff);
        update_overall(max_diff);

        $display("");
        $display("================ BLOCK-LEVEL CHECK ================");
        for (blk = 0; blk < 6; blk = blk + 1) begin
            report_first_n_mat_t_d($sformatf("block%0d_output", blk + 1), block_out_flat[blk], block_ref_flat[blk], SAMPLE_N, max_diff);
            update_overall(max_diff);
        end

        $display("[vit_pl_golden] overall_first_%0d_max_abs_diff=%0d", SAMPLE_N, overall_max_diff);

        $finish;
    end

endmodule

module tb_vit_pl_golden_basic_seed1234;
    tb_vit_pl_golden_basic_common #(.ROOT("../exports/seed1234")) run();
endmodule

module tb_vit_pl_golden_basic_seed2026;
    tb_vit_pl_golden_basic_common #(.ROOT("../exports/seed2026")) run();
endmodule
