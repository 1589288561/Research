# ViT Zynq Inference PL Golden Reference

## What This Project Is

This project is a Zynq-oriented FPGA golden-reference flow for Vision Transformer inference.

It validates a fixed-point PL forward path against a PyTorch reference model. The project is organized for review as a hardware-verification flow rather than as a model-training repository.

## What This Project Is Not

This project is not:

```text
a training accelerator
a Vivado block design
a board-ready AXI IP
a timing-closed FPGA implementation
```

It is a correctness package for the inference forward path.

## Upstream Python Reference

The PyTorch model and data-loading structure are adapted from:

```text
https://github.com/gejinchen/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10
```

The original repository provides the software ViT model and training-oriented project structure. This package keeps the model/data reference pieces and adds fixed-point export, PL RTL, and SystemVerilog verification.

## Repository Flow

```text
01_python_reference/
  PyTorch ViT model, data loader, and fixed checkpoints.

02_fixed_point_export/
  Python scripts plus generated Q4.12 fixed-point export payloads.

03_pl_golden_rtl/
  Verilog/SystemVerilog inference-forward PL golden model.

04_20_sample_test_vectors/
  Compact first-20-sample verification inputs and expected outputs.

05_systemverilog_testbench/
  Questa testbenches that load exported files and compare PL outputs.

06_verification_results/
  Verification notes and result-summary area.

07_docs_and_handoff/
  Boundary and handoff documentation.
```

## Numeric Format

The hardware payload uses signed Q4.12 fixed-point integers:

```text
WIDTH = 16
FRAC  = 12
```

`pred_class` is a categorical integer and is not Q4.12-scaled.

## PL Boundary

The PL golden model starts from:

```text
embedding_output
```

and computes:

```text
6 encoder blocks
encoder final LayerNorm
model LayerNorm
classifier fc1
tanh
classifier fc2
logits
pred_class
```

## Main RTL Files

```text
../03_pl_golden_rtl/vit_pl_golden_top.sv
../03_pl_golden_rtl/encoder_6block_with_2norm_top.sv
../03_pl_golden_rtl/encoder_block_top.v
../03_pl_golden_rtl/att_block_50x64.v
../03_pl_golden_rtl/att_core_50x64.v
../03_pl_golden_rtl/mlp_block_50x64.v
../03_pl_golden_rtl/classifier_head_64x10.v
```

## Exported Reference Payloads

Structured one-sample payloads:

```text
../02_fixed_point_export/exports/seed1234
../02_fixed_point_export/exports/seed2026
```

20-sample verification payloads:

```text
../04_20_sample_test_vectors/seed1234/fixed
../04_20_sample_test_vectors/seed2026/fixed
```

Each 20-sample folder contains:

```text
embedding_output_20.txt
logits_20.txt
pred_class_20.txt
label_20.txt
sample_count.txt
```

## Reproduce Exports

From the repository root:

```text
cd 02_fixed_point_export
python dump_vit_pl_two_seeds_structured.py
python dump_pl_io_all_samples.py
```

The dataset is downloaded by `torchvision` when needed and is ignored by Git through `.gitignore`.

## Run Questa Verification

From:

```text
05_systemverilog_testbench/
```

Compile:

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

Important output fields:

```text
checked_samples
pred_mismatch_count
overall_logits_max_abs_diff
PASS / MISMATCH per sample
```

## Current Engineering Status

Completed:

```text
PyTorch reference model
fixed checkpoints
Q4.12 export scripts
structured fixed-point payloads
20-sample verification vectors
PL forward golden RTL
SystemVerilog testbench
```

Not yet included:

```text
timed accelerator scheduling
AXI/BRAM integration
Vivado/Vitis project
board software
training/backpropagation hardware
```
