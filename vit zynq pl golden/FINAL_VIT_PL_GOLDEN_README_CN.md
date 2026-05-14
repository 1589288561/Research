# ViT Zynq PL Golden Reference 中文说明

## 1. 项目目标

本文件夹是一个面向 Zynq PS/PL 划分的 ViT-MNIST PL golden reference 工程。项目目标是用 Verilog/SystemVerilog 实现 ViT 推理流程中适合放入 PL 的主要计算部分，并用 Python 导出的 fixed-point reference 数据进行仿真验证。

本工程的软件模型代码参考自以下开源项目：

```text
GitHub reference:
https://github.com/gejinchen/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10

Upstream repository:
https://github.com/s-chh/PyTorch-Scratch-Vision-Transformer-ViT
```

其中 `model.py`、`data_loader.py`、`main.py`、`solver.py` 等 PyTorch 侧代码来自该类 ViT-MNIST/CIFAR10 scratch implementation。本工程在此软件模型基础上导出 fixed-point reference，并实现对应的 PL-side Verilog golden model。

本工程采用的系统划分如下：

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

也就是说，PS/Python 负责准备 `embedding_output`，PL/Verilog 负责从 `embedding_output` 一直计算到最终分类结果。这个边界适合后续上板学习，因为 PS 只需要向 PL 提供 embedding 输入，并从 PL 读回 10 维 `logits` 或 1 个 `pred_class`。

注意：当前工程是 golden reference 仿真工程，不是最终可上板的时序 accelerator。它主要用于确认 fixed-point 数据、PL 计算边界、encoder + classifier 计算结果是否正确。

## 2. 模型结构

本工程对应的 PyTorch 模型结构来自 `model.py` 中的 `VisionTransformer`：

```text
VisionTransformer.forward()
  -> embedding
  -> encoder
  -> norm
  -> classifier
```

其中本工程的 PL golden top 覆盖以下部分：

```text
encoder
norm
classifier
```

其中 encoder 包含：

```text
6 x Encoder block
1 x encoder final LayerNorm
1 x model LayerNorm
```

每个 Encoder block 的计算结构为：

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

classifier 的计算结构为：

```text
pre_classifier[CLS token]
  -> classifier fc1: 64 -> 64
  -> tanh
  -> classifier fc2: 64 -> 10
  -> logits
  -> argmax pred_class
```

## 3. 主要尺寸和定点格式

本工程使用 full-scale MNIST ViT 配置：

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

这些参数定义在：

```text
verilog/fixed_params.vh
```

关键 packed bus 宽度包括：

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

## 4. 文件夹结构

根目录主要文件：

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

主要目录：

```text
data/
exports/
pl_io_20_samples/
verilog/
```

其中：

```text
data/     = MNIST 数据
exports/  = Python 导出的 fixed/float reference 数据
pl_io_20_samples/ = 前 20 张测试图的 PL 输入和最终 reference 输出
verilog/  = PL golden RTL 和 testbench
```

## 5. Python 数据准备

### 5.1 Checkpoint

本工程使用两套固定 seed 的模型 checkpoint：

```text
model_encoder_seed1234.pt
model_encoder_seed2026.pt
```

这两套 checkpoint 用于生成可复现的 reference 数据。它们的用途是验证 RTL 是否与 Python reference 对齐，不表示模型一定是训练好的高精度分类器。

### 5.2 导出 reference 数据

运行：

```bash
python dump_vit_pl_two_seeds_structured.py
```

该脚本会导出：

```text
exports/seed1234/
exports/seed2026/
```

每个 seed 下包含：

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

关键输入输出文件：

```text
exports/seedXXXX/embedding_output/fixed/embedding_output.txt
exports/seedXXXX/pre_classifier/fixed/pre_classifier.txt
exports/seedXXXX/classifier/fixed/logits.txt
exports/seedXXXX/classifier/fixed/pred_class.txt
```

含义：

```text
embedding_output.txt = PL golden top 输入，shape = 50 x 64
pre_classifier.txt   = encoder + final norms 输出，shape = 50 x 64
logits.txt           = classifier 输出，shape = 10
pred_class.txt       = argmax 类别编号，不是 Q4.12 定点数
```

当前导出的 reference 预测结果：

```text
seed1234 pred_class = 0
seed2026 pred_class = 3
```

### 5.3 导出前 20 张图的 PL IO 数据

如果只想做多输入样本的端到端检查，而不需要中间 debug 数据，可以运行：

```bash
python dump_pl_io_all_samples.py
```

该脚本会导出每个 seed 的前 20 张 MNIST test sample：

```text
pl_io_20_samples/seed1234/fixed/
pl_io_20_samples/seed2026/fixed/
```

每个目录下包含：

```text
embedding_output_20.txt
logits_20.txt
pred_class_20.txt
label_20.txt
sample_count.txt
```

含义：

```text
embedding_output_20.txt = 20 张图的 PL 输入，每行一个 50 x 64 embedding_output
logits_20.txt           = 20 张图的 Python fixed reference logits，每行 10 个值
pred_class_20.txt       = 20 张图的 Python reference pred_class，每行 1 个类别编号
label_20.txt            = 20 张图的 MNIST ground-truth label，每行 1 个真实标签
sample_count.txt        = 样本数，当前为 20
```

注意：`label` 是数据集真实标签，`pred_class` 是模型预测类别。当前验证目标是 RTL 是否复现 Python reference，因此主要检查：

```text
RTL pred_class == Python reference pred_class
```

而不是检查：

```text
pred_class == label
```

## 6. Encoder 导出数据

每个 block 的 basic 数据目录：

```text
exports/seedXXXX/blockN/basic/fixed/
```

block 输入输出：

```text
block_input.txt
block_output.txt
```

block 参数：

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

每个 block 还保留 debug 数据：

```text
exports/seedXXXX/blockN/debug/fixed/
```

典型 debug 文件包括：

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

## 7. Final Norm 和 Classifier 导出数据

两个 final norm 的数据：

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

classifier 数据目录：

```text
exports/seedXXXX/classifier/fixed/
```

主要文件：

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

## 8. LUT 准备

本工程使用三个 LUT：

```text
verilog/exp_lut_q412_full.vh
verilog/gelu_lut_q412_full.vh
verilog/tanh_lut_q412_full.vh
```

用途：

```text
exp_lut_q412_full.vh  = attention softmax exp 近似
gelu_lut_q412_full.vh = MLP GELU 近似
tanh_lut_q412_full.vh = classifier tanh 近似
```

如果需要重新生成 tanh LUT：

```bash
python make_tanh_lut_from_torch.py
copy tanh_lut_q412_full.vh verilog\
```

## 9. Verilog 文件说明

核心 RTL 文件：

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

顶层模块：

```text
verilog/vit_pl_golden_top.sv
```

该顶层连接：

```text
embedding_output
  -> encoder_6block_with_2norm_top
  -> classifier_head_64x10
  -> logits / pred_class
```

classifier 模块：

```text
verilog/classifier_head_64x10.v
```

该模块实现：

```text
CLS token select
classifier fc1
tanh LUT
classifier fc2
argmax
```

## 10. Testbench 说明

主要 testbench：

```text
verilog/tb_vit_pl_golden_basic.sv
verilog/tb_vit_pl_20_samples.sv
```

公共 IO helper：

```text
verilog/tb_encoder_io.svh
```

testbench 会读取 fixed txt 文件，加载：

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

检查内容包括：

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

输出格式类似：

```text
idx expected actual diff
```

并报告：

```text
first_20_max_abs_diff
logits max_abs_diff
```

判断时不要只看 `pred_class`。`pred_class` 一致说明最终 argmax 一致；更严格的验证应同时观察 `logits` 和关键中间结果的 diff。

### 10.1 单张 debug testbench

`tb_vit_pl_golden_basic.sv` 用于检查单张图的完整 debug 链路。它会比较：

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

这个 testbench 适合定位误差来源。

### 10.2 前 20 张样本 testbench

`tb_vit_pl_20_samples.sv` 用于检查前 20 张图的端到端 PL 输出。它不读取中间 debug 文件，只读取：

```text
pl_io_20_samples/seedXXXX/fixed/embedding_output_20.txt
pl_io_20_samples/seedXXXX/fixed/logits_20.txt
pl_io_20_samples/seedXXXX/fixed/pred_class_20.txt
pl_io_20_samples/seedXXXX/fixed/label_20.txt
pl_io_20_samples/seedXXXX/fixed/sample_count.txt
```

每张图会输出类似：

```text
PASS sample=0 label=7 pred=0 logits_max_diff=10
```

含义：

```text
PASS            = RTL pred_class 与 Python reference pred_class 一致
sample          = 样本编号，从 0 开始
label           = MNIST ground-truth label
pred            = RTL / reference 一致的预测类别
logits_max_diff = 当前样本 10 个 logits 中 RTL 与 reference 的最大绝对误差
```

如果预测类别不一致，会输出：

```text
MISMATCH sample=... label=... expected_pred=... actual_pred=... logits_max_diff=...
```

最后会输出 summary：

```text
checked_samples=20
pred_mismatch_count=0
overall_logits_max_abs_diff=...
[20_samples] PASS: all predicted classes match reference.
```

其中：

```text
pred_mismatch_count=0
```

是前 20 张样本最终分类结果全部对齐 reference 的主要通过条件。

## 11. Questa 运行命令

建议从 Verilog 目录启动仿真：

```bash
cd "C:\Users\pzr\Desktop\summer research\vit zynq pl golden\verilog"
```

编译：

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

运行单张 debug seed1234：

```tcl
vsim -c tb_vit_pl_golden_basic_seed1234
run -all
```

运行单张 debug seed2026：

```tcl
vsim -c tb_vit_pl_golden_basic_seed2026
run -all
```

运行前 20 张样本 seed1234：

```tcl
vsim -c tb_vit_pl_20_samples_seed1234
run -all
```

运行前 20 张样本 seed2026：

```tcl
vsim -c tb_vit_pl_20_samples_seed2026
run -all
```

testbench 默认路径为：

```text
../exports/seed1234
../exports/seed2026
```

因此建议从 `verilog` 目录运行 Questa。

## 12. 当前验证状态

当前 testbench 已在 Questa 中观察到最终预测一致：

```text
seed1234 expected pred_class = 0, RTL actual pred_class = 0
seed2026 expected pred_class = 3, RTL actual pred_class = 3
```

前 20 张样本测试中，已观察到逐样本输出均为 `PASS`。示例输出：

```text
PASS sample=0 label=7 pred=0 logits_max_diff=10
PASS sample=1 label=2 pred=0 logits_max_diff=9
PASS sample=2 label=1 pred=0 logits_max_diff=10
...
PASS sample=19 label=4 pred=0 logits_max_diff=11
```

这说明前 20 张图中，RTL 的 `pred_class` 与 Python fixed reference 的 `pred_class` 全部一致。`label` 与 `pred` 不一致不代表 RTL 错误，因为本工程验证的是硬件是否复现 reference，而不是模型分类准确率。

推荐完整验收标准：

```text
1. pred_class 与 reference 一致
2. logits max_abs_diff 在可接受范围内
3. pre_classifier first_20_max_abs_diff 不异常
4. classifier_fc1 / classifier_tanh diff 不异常
5. block1~block6 的 first_20_max_abs_diff 没有突然失控
```

## 13. 后续上板重构方向

当前 Verilog 是 golden reference 风格，主要特点是：

```text
大 packed bus
组合逻辑计算
testbench 从 txt 文件读取数据
没有 clk/reset/start/done
没有 AXI/BRAM/DDR 接口
```

后续如果要面向 Zynq 上板，需要逐步重构为时序 accelerator：

```text
clk / reset / start / done
BRAM 或 AXI 数据接口
可复用 MAC engine
分阶段 FSM
中间结果 buffer
可综合、可过 timing 的时序结构
```

推荐从较小模块开始，例如：

```text
classifier
单个 FC layer
通用 MAC engine
Q/K/V linear projection
single encoder block
```

这样可以在保持 golden reference 对照的同时，逐步学习和实现真正可上板的 PL accelerator。
