# 05 SystemVerilog Testbench

## Purpose

This step verifies the PL golden RTL against fixed-point reference files generated from the PyTorch model.

## Files

```text
tb_vit_pl_20_samples.sv
```

Main end-to-end testbench. It loads 20-sample embeddings, expected logits, expected predicted classes, and labels for both seeds.

```text
tb_vit_pl_golden_basic.sv
```

Single-sample/debug-oriented testbench for checking the structured export path.

```text
tb_encoder_io.svh
```

Shared file-loading helper tasks for weights, activations, and fixed-point vectors.

## How To Run

Compile the RTL and testbench in Questa/ModelSim:

```text
vlog +incdir+../03_pl_golden_rtl ../03_pl_golden_rtl/*.v ../03_pl_golden_rtl/*.sv tb_vit_pl_20_samples.sv
```

Run seed 1234:

```text
vsim tb_vit_pl_20_samples_seed1234
run -all
```

Run seed 2026:

```text
vsim tb_vit_pl_20_samples_seed2026
run -all
```

## Inputs

Weights and intermediate reference files:

```text
../02_fixed_point_export/exports/seed1234/
../02_fixed_point_export/exports/seed2026/
```

20-sample input/output vectors:

```text
../04_20_sample_test_vectors/seed1234/fixed/
../04_20_sample_test_vectors/seed2026/fixed/
```

## Expected Results

The testbench prints per-sample status and summary fields:

```text
PASS / MISMATCH
checked_samples
pred_mismatch_count
overall_logits_max_abs_diff
```

`pred_mismatch_count = 0` means the PL predicted classes match the Python reference for the checked samples.

## Result Of This Step

```text
RTL execution + reference comparison -> evidence that the PL golden model matches the Python export
```
