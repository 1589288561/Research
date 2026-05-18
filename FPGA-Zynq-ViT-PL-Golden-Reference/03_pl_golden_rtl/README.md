# 03 PL Golden RTL

## Purpose

This step contains the Verilog/SystemVerilog golden model for the PL-side ViT inference forward path.

## Files

Top-level and composition:

```text
vit_pl_golden_top.sv
encoder_6block_with_2norm_top.sv
encoder_block_top.v
```

Attention path:

```text
att_block_50x64.v
att_core_50x64.v
exp_lut_q412_full.vh
```

MLP and residual path:

```text
mlp_block_50x64.v
fc1_stage_50x64_to_50x128.v
gelu_stage_50x128.v
gelu_lut_q412_full.vh
fc2_stage_50x128_to_50x64.v
residual_add_50x64.v
```

Normalization and classifier:

```text
norm_stage_50x64.v
classifier_head_64x10.v
tanh_lut_q412_full.vh
```

Shared fixed-point definitions:

```text
fixed_params.vh
```

## Computation Boundary

```text
embedding_output
  -> 6 encoder blocks
  -> encoder final LayerNorm
  -> model LayerNorm
  -> classifier
  -> logits / pred_class
```

## How This Step Is Used

The testbenches in `../05_systemverilog_testbench/` compile these RTL files, load fixed-point reference data from `../02_fixed_point_export/`, and compare the PL golden outputs against Python-generated reference outputs.

## Result Of This Step

```text
fixed-point PL forward model -> RTL outputs for logits and pred_class comparison
```
