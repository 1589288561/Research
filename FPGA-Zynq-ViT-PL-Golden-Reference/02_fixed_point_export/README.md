# 02 Fixed-Point Export

## Purpose

This step converts the PyTorch ViT reference into hardware-readable fixed-point files for RTL simulation.

## Files

```text
dump_vit_pl_two_seeds_structured.py
```

Exports detailed per-layer payloads for `seed1234` and `seed2026`, including block inputs/outputs, weights, biases, LayerNorm values, classifier values, logits, and predicted classes.

```text
dump_pl_io_all_samples.py
```

Exports the compact 20-sample end-to-end verification payload used by the SystemVerilog testbench.

```text
exports/
```

Generated structured reference payloads. Each seed includes:

```text
embedding_output/
block1/ ... block6/
encoder_final_norm/
model_norm/
pre_classifier/
classifier/
```

## Numeric Format

Most tensor values are exported as signed Q4.12 fixed-point integers:

```text
WIDTH = 16
FRAC = 12
```

`pred_class` is a categorical class ID and is stored as an integer, not as a Q4.12-scaled value.

## How To Run

From this folder:

```text
python dump_vit_pl_two_seeds_structured.py
python dump_pl_io_all_samples.py
```

## Outputs

Structured exports:

```text
exports/seed1234/
exports/seed2026/
```

20-sample vectors generated into:

```text
../04_20_sample_test_vectors/seed1234/fixed/
../04_20_sample_test_vectors/seed2026/fixed/
```

## Result Of This Step

```text
PyTorch inference tensors -> fixed-point reference files for RTL/testbench comparison
```
