# Zynq PS/PL Golden Boundary

## What This Document Defines

This project uses a Zynq-style PS/PL split for inference-forward verification.

The goal is to keep image/data preparation on the Python/PS side and verify the main ViT forward computation on the PL side.

## Boundary

PS/Python prepares:

```text
MNIST image
  -> preprocessing
  -> patch embedding
  -> embedding_output
```

PL/SystemVerilog computes:

```text
embedding_output
  -> 6 encoder blocks
  -> encoder final LayerNorm
  -> model LayerNorm
  -> classifier fc1
  -> tanh
  -> classifier fc2
  -> logits / pred_class
```

## Why This Boundary

This boundary keeps the high-volume ViT forward path in PL while keeping the input-preparation side simple.

Keeping the classifier in PL means the software side only needs to read back:

```text
10 logits
```

or:

```text
1 pred_class
```

It does not need to pull back a full `50 x 64` intermediate feature map for final classification.

## Payload Locations

Structured fixed-point exports:

```text
../02_fixed_point_export/exports/seed1234
../02_fixed_point_export/exports/seed2026
```

20-sample verification vectors:

```text
../04_20_sample_test_vectors/seed1234/fixed
../04_20_sample_test_vectors/seed2026/fixed
```

## RTL Entry Points

PL golden top:

```text
../03_pl_golden_rtl/vit_pl_golden_top.sv
```

20-sample verification testbench:

```text
../05_systemverilog_testbench/tb_vit_pl_20_samples.sv
```

## Current Limit

This is still a golden reference simulation project.

It does not include:

```text
AXI interface
BRAM/DDR controller
start/done control
timing closure
Vitis software driver
board deployment
training/backpropagation
```

The next architecture step would be to refactor the large forward-path modules into timed MAC engines and memory-backed stages while keeping this golden model as the correctness reference.
