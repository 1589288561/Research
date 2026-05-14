# ViT Zynq PL Golden Reference

## 1. Project Goal

This folder is a ViT-MNIST PL golden reference project designed around a Zynq PS/PL partition. The goal is to implement the main PL-side inference computation in Verilog/SystemVerilog and verify it against fixed-point reference data exported from Python.

The software model code in this project is based on the following open-source repositories:

```text
GitHub reference:
https://github.com/gejinchen/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10

Upstream repository:
https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT
```

The PyTorch-side files such as `model.py`, `data_loader.py`, `main.py`, and `solver.py` come from this style of ViT-MNIST/CIFAR10 scratch implementation. This project exports fixed-point reference data from that software model and implements the corresponding PL-side Verilog golden model.

The project uses the following system boundary:

```text
PS / Python prepares:
  image -> embedding_output

PL / Verilog computes:
  embedding_output
    -> encoder block 1
    -> encoder block 2
    -> encoder block 3
    -> encoder block 4
    -> encoder block 5
    -> encoder block 6
    -> encoder final LayerNorm
    -> VisionTransformer.norm
    -> classifier fc1
    -> tanh
    -> classifier fc2
    -> logits
    -> pred_class
```

In this boundary, PS/Python prepares `embedding_output`, while PL/Verilog computes from `embedding_output` all the way to the final classification result. This is a practical boundary for later board work because PS only needs to provide the embedding input and read back either 10 `logits` values or one `pred_class`.

Note: this project is a golden reference simulation project. It is not yet a timed board-ready accelerator. Its purpose is to validate the fixed-point data path, the PL computation boundary, and the encoder + classifier RTL results.

## 2. Model Structure

The PyTorch model is the `VisionTransformer` defined in `model.py`:

```text
VisionTransformer.forward()
  -> embedding
  -> encoder
  -> norm
  -> classifier
```

The PL golden top covers:

```text
encoder
norm
classifier
```

The encoder contains:

```text
6 x Encoder block
1 x encoder final LayerNorm
1 x model LayerNorm
```

Each Encoder block computes:

```text
x
  -> norm1
  -> multi-head self-attention
  -> residual add
  -> norm2
  -> fc1
  -> GELU
  -> fc2
  -> residual add
  -> y
```

The classifier computes:

```text
pre_classifier[CLS token]
  -> classifier fc1: 64 -> 64
  -> tanh
  -> classifier fc2: 64 -> 10
  -> logits
  -> argmax pred_class
```

## 3. Dimensions and Fixed-Point Format

This project uses the full-scale MNIST ViT configuration:

```text
T = 50 tokens
D = 64 embedding channels
HEADS = 4
HEAD_DIM = 16
DFF = 128
N_CLASSES = 10
WIDTH = 16
FRAC = 12
fixed-point format = Q4.12
```

These parameters are defined in:

```text
verilog/fixed_params.vh
```

Important packed bus widths include:

```text
MAT_T_D_W   = 50 x 64 x 16-bit
MAT_T_DFF_W = 50 x 128 x 16-bit
VEC_D_W     = 64 x 16-bit
VEC_DFF_W   = 128 x 16-bit
VEC_C_W     = 10 x 16-bit
W_D_D_W     = 64 x 64 x 16-bit
W_DFF_D_W   = 128 x 64 x 16-bit
W_D_DFF_W   = 64 x 128 x 16-bit
W_C_D_W     = 10 x 64 x 16-bit
```

## 4. Folder Structure

Main files in the project root:

```text
model.py
data_loader.py
model_encoder_seed1234.pt
model_encoder_seed2026.pt
dump_vit_pl_two_seeds_structured.py
dump_pl_io_all_samples.py
make_tanh_lut_from_torch.py
README_PL_BOUNDARY.md
FINAL_VIT_PL_GOLDEN_README_CN.md
FINAL_VIT_PL_GOLDEN_README_EN.md
```

Main directories:

```text
data/
exports/
pl_io_20_samples/
verilog/
```

Meaning:

```text
data/     = MNIST data
exports/  = fixed/float reference data exported from Python
pl_io_20_samples/ = PL input and final reference output for the first 20 test images
verilog/  = PL golden RTL and testbench
```

## 5. Python Data Preparation

### 5.1 Checkpoints

This project uses two fixed-seed model checkpoints:

```text
model_encoder_seed1234.pt
model_encoder_seed2026.pt
```

These checkpoints generate reproducible reference data. They are used to verify that RTL matches the Python reference. They do not imply that the model is a trained high-accuracy classifier.

### 5.2 Export Reference Data

Run:

```bash
python dump_vit_pl_two_seeds_structured.py
```

The script exports:

```text
exports/seed1234/
exports/seed2026/
```

Each seed directory contains:

```text
input_debug/
embedding_output/
block1/
block2/
block3/
block4/
block5/
block6/
encoder_final_norm/
model_norm/
pre_classifier/
classifier/
```

Key input/output files:

```text
exports/seedXXXX/embedding_output/fixed/embedding_output.txt
exports/seedXXXX/pre_classifier/fixed/pre_classifier.txt
exports/seedXXXX/classifier/fixed/logits.txt
exports/seedXXXX/classifier/fixed/pred_class.txt
```

Meaning:

```text
embedding_output.txt = PL golden top input, shape = 50 x 64
pre_classifier.txt   = encoder + final norms output, shape = 50 x 64
logits.txt           = classifier output, shape = 10
pred_class.txt       = argmax class id, not a Q4.12 fixed-point value
```

Current exported reference predictions:

```text
seed1234 pred_class = 0
seed2026 pred_class = 3
```

### 5.3 Export PL IO Data for the First 20 Images

For a lightweight multi-input end-to-end check without intermediate debug files, run:

```bash
python dump_pl_io_all_samples.py
```

The script exports the first 20 MNIST test samples for each seed:

```text
pl_io_20_samples/seed1234/fixed/
pl_io_20_samples/seed2026/fixed/
```

Each directory contains:

```text
embedding_output_20.txt
logits_20.txt
pred_class_20.txt
label_20.txt
sample_count.txt
```

Meaning:

```text
embedding_output_20.txt = PL inputs for 20 images, one 50 x 64 embedding_output per line
logits_20.txt           = Python fixed reference logits for 20 images, 10 values per line
pred_class_20.txt       = Python reference pred_class values, one class id per line
label_20.txt            = MNIST ground-truth labels, one label per line
sample_count.txt        = sample count, currently 20
```

Note: `label` is the dataset ground-truth label, while `pred_class` is the model prediction. The verification target is:

```text
RTL pred_class == Python reference pred_class
```

not:

```text
pred_class == label
```

## 6. Encoder Export Data

Each block has a basic data directory:

```text
exports/seedXXXX/blockN/basic/fixed/
```

Block input/output:

```text
block_input.txt
block_output.txt
```

Block parameters:

```text
norm1_weight.txt
norm1_bias.txt
wq_weight.txt
wq_bias.txt
wk_weight.txt
wk_bias.txt
wv_weight.txt
wv_bias.txt
norm2_weight.txt
norm2_bias.txt
fc1_weight.txt
fc1_bias.txt
fc2_weight.txt
fc2_bias.txt
```

Each block also keeps debug data:

```text
exports/seedXXXX/blockN/debug/fixed/
```

Typical debug files include:

```text
norm1_output.txt
q_output.txt
k_output.txt
v_output.txt
score_output_head0.txt
score_output_head1.txt
score_output_head2.txt
score_output_head3.txt
prob_output_head0.txt
prob_output_head1.txt
prob_output_head2.txt
prob_output_head3.txt
attention_only_output.txt
att_block_output.txt
norm2_output.txt
fc1_output.txt
gelu_output.txt
fc2_output.txt
mlp_block_output.txt
```

## 7. Final Norm and Classifier Export Data

Final norm data:

```text
exports/seedXXXX/encoder_final_norm/fixed/norm_input.txt
exports/seedXXXX/encoder_final_norm/fixed/norm_output.txt
exports/seedXXXX/encoder_final_norm/fixed/norm_weight.txt
exports/seedXXXX/encoder_final_norm/fixed/norm_bias.txt

exports/seedXXXX/model_norm/fixed/norm_input.txt
exports/seedXXXX/model_norm/fixed/norm_output.txt
exports/seedXXXX/model_norm/fixed/norm_weight.txt
exports/seedXXXX/model_norm/fixed/norm_bias.txt
```

Classifier data directory:

```text
exports/seedXXXX/classifier/fixed/
```

Main files:

```text
cls_input.txt
classifier_fc1_weight.txt
classifier_fc1_bias.txt
classifier_fc1_output.txt
classifier_tanh_output.txt
classifier_fc2_weight.txt
classifier_fc2_bias.txt
logits.txt
pred_class.txt
```

## 8. LUT Preparation

This project uses three LUT files:

```text
verilog/exp_lut_q412_full.vh
verilog/gelu_lut_q412_full.vh
verilog/tanh_lut_q412_full.vh
```

Purpose:

```text
exp_lut_q412_full.vh  = approximation for attention softmax exp
gelu_lut_q412_full.vh = approximation for MLP GELU
tanh_lut_q412_full.vh = approximation for classifier tanh
```

To regenerate the tanh LUT:

```bash
python make_tanh_lut_from_torch.py
copy tanh_lut_q412_full.vh verilog\
```

## 9. Verilog Files

Core RTL files:

```text
verilog/vit_pl_golden_top.sv
verilog/encoder_6block_with_2norm_top.sv
verilog/encoder_block_top.v
verilog/att_block_50x64.v
verilog/att_core_50x64.v
verilog/mlp_block_50x64.v
verilog/norm_stage_50x64.v
verilog/residual_add_50x64.v
verilog/fc1_stage_50x64_to_50x128.v
verilog/gelu_stage_50x128.v
verilog/fc2_stage_50x128_to_50x64.v
verilog/classifier_head_64x10.v
```

Top module:

```text
verilog/vit_pl_golden_top.sv
```

The top module connects:

```text
embedding_output
  -> encoder_6block_with_2norm_top
  -> classifier_head_64x10
  -> logits / pred_class
```

Classifier module:

```text
verilog/classifier_head_64x10.v
```

This module implements:

```text
CLS token select
classifier fc1
tanh LUT
classifier fc2
argmax
```

## 10. Testbench

Main testbench:

```text
verilog/tb_vit_pl_golden_basic.sv
verilog/tb_vit_pl_20_samples.sv
```

Shared IO helper:

```text
verilog/tb_encoder_io.svh
```

The testbench reads fixed txt files and loads:

```text
embedding input
encoder block weights
final norm weights
classifier weights
reference pre_classifier
reference classifier intermediate outputs
reference logits
reference pred_class
```

Checked outputs:

```text
logits
pred_class
pre_classifier
cls_input
classifier_fc1
classifier_tanh
block1_output
block2_output
block3_output
block4_output
block5_output
block6_output
```

The output format is similar to:

```text
idx expected actual diff
```

The testbench also reports:

```text
first_20_max_abs_diff
logits max_abs_diff
```

Do not only check `pred_class`. Matching `pred_class` means the final argmax is correct, but stronger validation should also inspect `logits` and key intermediate diffs.

### 10.1 Single-Image Debug Testbench

`tb_vit_pl_golden_basic.sv` checks the full debug path for one image. It compares:

```text
logits
pred_class
pre_classifier
cls_input
classifier_fc1
classifier_tanh
block1_output
block2_output
block3_output
block4_output
block5_output
block6_output
```

This testbench is useful for locating where an error starts.

### 10.2 First-20-Sample Testbench

`tb_vit_pl_20_samples.sv` checks end-to-end PL outputs for the first 20 images. It does not read intermediate debug files. It only reads:

```text
pl_io_20_samples/seedXXXX/fixed/embedding_output_20.txt
pl_io_20_samples/seedXXXX/fixed/logits_20.txt
pl_io_20_samples/seedXXXX/fixed/pred_class_20.txt
pl_io_20_samples/seedXXXX/fixed/label_20.txt
pl_io_20_samples/seedXXXX/fixed/sample_count.txt
```

Each image prints a line like:

```text
PASS sample=0 label=7 pred=0 logits_max_diff=10
```

Meaning:

```text
PASS            = RTL pred_class matches Python reference pred_class
sample          = sample index, starting from 0
label           = MNIST ground-truth label
pred            = matching RTL/reference predicted class
logits_max_diff = maximum absolute difference among the 10 logits for this sample
```

If the predicted class does not match, the testbench prints:

```text
MISMATCH sample=... label=... expected_pred=... actual_pred=... logits_max_diff=...
```

The final summary looks like:

```text
checked_samples=20
pred_mismatch_count=0
overall_logits_max_abs_diff=...
[20_samples] PASS: all predicted classes match reference.
```

The main pass condition for the first 20 samples is:

```text
pred_mismatch_count=0
```

## 11. Questa Commands

Start simulation from the Verilog directory:

```bash
cd "C:\Users\pzr\Desktop\summer research\vit zynq pl golden\verilog"
```

Compile:

```tcl
vlib work
vlog -sv +incdir+. fixed_params.vh
vlog -sv +incdir+. att_core_50x64.v
vlog -sv +incdir+. att_block_50x64.v
vlog -sv +incdir+. norm_stage_50x64.v
vlog -sv +incdir+. residual_add_50x64.v
vlog -sv +incdir+. fc1_stage_50x64_to_50x128.v
vlog -sv +incdir+. gelu_stage_50x128.v
vlog -sv +incdir+. fc2_stage_50x128_to_50x64.v
vlog -sv +incdir+. mlp_block_50x64.v
vlog -sv +incdir+. encoder_block_top.v
vlog -sv +incdir+. encoder_6block_with_2norm_top.sv
vlog -sv +incdir+. classifier_head_64x10.v
vlog -sv +incdir+. vit_pl_golden_top.sv
vlog -sv +incdir+. tb_vit_pl_golden_basic.sv
vlog -sv +incdir+. tb_vit_pl_20_samples.sv
```

Run single-image debug seed1234:

```tcl
vsim -c tb_vit_pl_golden_basic_seed1234
run -all
```

Run single-image debug seed2026:

```tcl
vsim -c tb_vit_pl_golden_basic_seed2026
run -all
```

Run first-20-sample seed1234:

```tcl
vsim -c tb_vit_pl_20_samples_seed1234
run -all
```

Run first-20-sample seed2026:

```tcl
vsim -c tb_vit_pl_20_samples_seed2026
run -all
```

The default testbench paths are:

```text
../exports/seed1234
../exports/seed2026
```

Therefore, running Questa from the `verilog` directory is recommended.

## 12. Current Verification Status

The final predictions have been observed to match in Questa:

```text
seed1234 expected pred_class = 0, RTL actual pred_class = 0
seed2026 expected pred_class = 3, RTL actual pred_class = 3
```

In the first-20-sample test, all observed per-sample lines are `PASS`. Example output:

```text
PASS sample=0 label=7 pred=0 logits_max_diff=10
PASS sample=1 label=2 pred=0 logits_max_diff=9
PASS sample=2 label=1 pred=0 logits_max_diff=10
...
PASS sample=19 label=4 pred=0 logits_max_diff=11
```

This means RTL `pred_class` matches the Python fixed reference `pred_class` for all first 20 images. A mismatch between `label` and `pred` does not indicate an RTL error, because this project verifies hardware reproduction of the reference, not model classification accuracy.

Recommended full acceptance checks:

```text
1. pred_class matches the reference
2. logits max_abs_diff is acceptable
3. pre_classifier first_20_max_abs_diff is not abnormal
4. classifier_fc1 / classifier_tanh diffs are not abnormal
5. block1~block6 first_20_max_abs_diff values do not suddenly diverge
```

## 13. Direction for Board-Oriented Refactoring

The current Verilog is a golden reference implementation. Its main traits are:

```text
large packed buses
combinational computation
testbench txt file loading
no clk/reset/start/done
no AXI/BRAM/DDR interface
```

For Zynq board deployment, the design needs to be refactored into a timed accelerator:

```text
clk / reset / start / done
BRAM or AXI data interface
reusable MAC engine
staged FSM
intermediate result buffers
synthesizable timing-friendly structure
```

Recommended starting points:

```text
classifier
single FC layer
generic MAC engine
Q/K/V linear projection
single encoder block
```

This allows the timed accelerator to be developed step by step while keeping the golden reference available for correctness checking.
