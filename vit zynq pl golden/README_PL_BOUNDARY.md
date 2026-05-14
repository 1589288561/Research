# Zynq PS/PL Golden Boundary

This project keeps the same fixed-point RTL style as the earlier golden
reference projects, but changes the boundary to match a practical first Zynq
accelerator.

## Boundary

PS prepares:

```text
MNIST image -> quantization / embedding_output
```

PL computes:

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

This is a good first board-oriented split because the encoder is the large
compute load and the classifier is small. Keeping the classifier in PL also
means PS only needs to read back 10 logits or one class id instead of a 50 x 64
feature map.

## Generate payloads

```bash
python dump_vit_pl_two_seeds_structured.py
python make_tanh_lut_from_torch.py
copy tanh_lut_q412_full.vh verilog\
```

The `exports/seedXXXX/classifier` folder contains classifier weights, debug
activations, logits, and predicted class.

## Questa source set

Compile the files in `verilog` with SystemVerilog enabled. The top-level
golden PL module is:

```text
vit_pl_golden_top.sv
```

The basic end-to-end PL golden testbench is:

```text
tb_vit_pl_golden_basic.sv
```

This is still a golden reference, not the final timed accelerator. The next
architecture step is to replace the large combinational layers with
clk/reset/start/done controlled MAC engines and memories.
