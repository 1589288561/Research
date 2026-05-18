`ifndef TB_ENCODER_IO_SVH
`define TB_ENCODER_IO_SVH

task automatic load_vec_d(input string path, output reg signed [`VEC_D_W-1:0] flat);
    integer fd, r, val, idx;
    begin
        flat = 0;
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("ERROR: cannot open %s", path);
            $finish;
        end
        for (idx = 0; idx < `D; idx = idx + 1) begin
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

task automatic load_vec_dff(input string path, output reg signed [`VEC_DFF_W-1:0] flat);
    integer fd, r, val, idx;
    begin
        flat = 0;
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("ERROR: cannot open %s", path);
            $finish;
        end
        for (idx = 0; idx < `DFF; idx = idx + 1) begin
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

task automatic load_mat_t_d(input string path, output reg signed [`MAT_T_D_W-1:0] flat);
    integer fd, r, val, idx;
    begin
        flat = 0;
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("ERROR: cannot open %s", path);
            $finish;
        end
        for (idx = 0; idx < `T * `D; idx = idx + 1) begin
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

task automatic load_mat_t_dff(input string path, output reg signed [`MAT_T_DFF_W-1:0] flat);
    integer fd, r, val, idx;
    begin
        flat = 0;
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("ERROR: cannot open %s", path);
            $finish;
        end
        for (idx = 0; idx < `T * `DFF; idx = idx + 1) begin
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

task automatic load_w_d_d(input string path, output reg signed [`W_D_D_W-1:0] flat);
    integer fd, r, val, idx;
    begin
        flat = 0;
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("ERROR: cannot open %s", path);
            $finish;
        end
        for (idx = 0; idx < `D * `D; idx = idx + 1) begin
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

task automatic load_w_dff_d(input string path, output reg signed [`W_DFF_D_W-1:0] flat);
    integer fd, r, val, idx;
    begin
        flat = 0;
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("ERROR: cannot open %s", path);
            $finish;
        end
        for (idx = 0; idx < `DFF * `D; idx = idx + 1) begin
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

task automatic load_w_d_dff(input string path, output reg signed [`W_D_DFF_W-1:0] flat);
    integer fd, r, val, idx;
    begin
        flat = 0;
        fd = $fopen(path, "r");
        if (fd == 0) begin
            $display("ERROR: cannot open %s", path);
            $finish;
        end
        for (idx = 0; idx < `D * `DFF; idx = idx + 1) begin
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

task automatic report_cmp_mat_t_d(
    input string name,
    input signed [`MAT_T_D_W-1:0] actual,
    input signed [`MAT_T_D_W-1:0] expected,
    output integer max_abs_diff
);
    integer i, diff;
    begin
        max_abs_diff = 0;
        for (i = 0; i < `T * `D; i = i + 1) begin
            diff = $signed(`ELEM(actual, i)) - $signed(`ELEM(expected, i));
            if (diff < 0) diff = -diff;
            if (diff > max_abs_diff) max_abs_diff = diff;
        end
        $display("[%s] max_abs_diff=%0d", name, max_abs_diff);
    end
endtask

task automatic report_cmp_mat_t_dff(
    input string name,
    input signed [`MAT_T_DFF_W-1:0] actual,
    input signed [`MAT_T_DFF_W-1:0] expected,
    output integer max_abs_diff
);
    integer i, diff;
    begin
        max_abs_diff = 0;
        for (i = 0; i < `T * `DFF; i = i + 1) begin
            diff = $signed(`ELEM(actual, i)) - $signed(`ELEM(expected, i));
            if (diff < 0) diff = -diff;
            if (diff > max_abs_diff) max_abs_diff = diff;
        end
        $display("[%s] max_abs_diff=%0d", name, max_abs_diff);
    end
endtask

task automatic print_diag_mat_t_d(
    input string name,
    input signed [`MAT_T_D_W-1:0] actual,
    input signed [`MAT_T_D_W-1:0] expected
);
    integer row, col, idx, expected_val, actual_val, diff;
    begin
        $display("======== %s diagonal compare ========", name);
        $display("    n   expected   actual   diff");
        for (row = 0; row < `T; row = row + 1) begin
            col = row;
            idx = row * `D + col;
            expected_val = $signed(`ELEM(expected, idx));
            actual_val = $signed(`ELEM(actual, idx));
            diff = actual_val - expected_val;
            $display("%5d %10d %8d %7d", row + 1, expected_val, actual_val, diff);
        end
    end
endtask

task automatic report_first_n_mat_t_d(
    input string name,
    input signed [`MAT_T_D_W-1:0] actual,
    input signed [`MAT_T_D_W-1:0] expected,
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

`endif
