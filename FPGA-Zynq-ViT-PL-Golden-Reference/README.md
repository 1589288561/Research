# FPGA-Zynq-ViT-Inference-PL-Golden-Reference

## Project Purpose

This repository is a Zynq/FPGA-oriented golden-reference flow for the Vision Transformer (ViT) inference forward path.

It starts from a PyTorch ViT reference, exports fixed-point Q4.12 hardware payloads, implements a Verilog/SystemVerilog PL-side golden model, and verifies the PL output against Python-generated reference data.

This is an inference-forward correctness project. It is not a training accelerator, not a Vivado block design, and not yet a board-ready AXI/BRAM implementation.

## What Is Included

```text
01_python_reference/
  PyTorch ViT model, dataset loader, and fixed checkpoints.

02_fixed_point_export/
  Python scripts and generated fixed-point exports for the model internals.

03_pl_golden_rtl/
  Verilog/SystemVerilog RTL for the PL-side inference golden model.

04_20_sample_test_vectors/
  Compact 20-sample input/output vectors for fast end-to-end verification.

05_systemverilog_testbench/
  Questa/SystemVerilog testbenches that drive RTL and compare outputs.

06_verification_results/
  Summary area for verification scope, expected result fields, and logs.

07_docs_and_handoff/
  Project boundary, methodology, and engineering handoff notes.

scripts/
  Helper scripts for regenerated lookup tables or packaging support.

assets/
  Reserved for diagrams or screenshots used by documentation.
```

## End-To-End Flow

```text
PyTorch ViT reference
  -> fixed checkpoints
  -> Q4.12 export scripts
  -> fixed-point TXT/PT payloads
  -> PL golden RTL
  -> SystemVerilog testbench
  -> logits / pred_class comparison
```

## Architecture Boundary

The current hardware boundary starts after patch embedding:

```text
Python / PS side:
  image preprocessing
  patch embedding
  embedding_output generation

PL / RTL side:
  6 ViT encoder blocks
  encoder final LayerNorm
  model LayerNorm
  classifier
  logits
  pred_class
```

The PL model is a correctness reference for the forward path. A later timed accelerator would add clocked scheduling, memories, AXI/BRAM interfaces, and board software.

## Verification Target

The compact verification path uses the first 20 MNIST test samples for two fixed checkpoint seeds:

```text
seed1234
seed2026
```

The testbench compares:

```text
PL logits      vs. Python fixed-point logits
PL pred_class  vs. Python categorical pred_class
```

Expected result fields:

```text
checked_samples
pred_mismatch_count
overall_logits_max_abs_diff
PASS / MISMATCH per sample
```

## Reproduce The Export Flow

Install Python dependencies:

```text
pip install -r requirements.txt
```

Regenerate the structured fixed-point payloads:

```text
cd 02_fixed_point_export
python dump_vit_pl_two_seeds_structured.py
python dump_pl_io_all_samples.py
```

`torchvision` downloads MNIST into `data/` when needed. That folder is ignored by Git.

## Run RTL Verification

From:

```text
05_systemverilog_testbench/
```

Compile in Questa/ModelSim:

```text
vlog +incdir+../03_pl_golden_rtl ../03_pl_golden_rtl/*.v ../03_pl_golden_rtl/*.sv tb_vit_pl_20_samples.sv
```

Run:

```text
vsim tb_vit_pl_20_samples_seed1234
run -all

vsim tb_vit_pl_20_samples_seed2026
run -all
```

## Python Reference Lineage

The Python model and dataset-loading structure are adapted from:

```text
https://github.com/gejinchen/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10
```

This repository keeps the useful PyTorch reference components and adds the hardware-facing pieces:

```text
fixed-point export
hardware-readable payloads
PL golden RTL
SystemVerilog verification
```

## Current Status

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

Not included yet:

```text
training / backpropagation hardware
timed accelerator scheduling
AXI / BRAM / DDR interface
Vivado / Vitis project
board deployment
```
