# 01 Python Reference

## Purpose

This step defines the floating-point Vision Transformer reference used as the source of truth for later fixed-point export and RTL verification.

It is adapted from:

```text
https://github.com/gejinchen/PyTorch-Vision-Transformer-ViT-MNIST-CIFAR10
```

## Files

```text
model.py
```

Defines the PyTorch ViT architecture: patch embedding, class token, positional embedding, self-attention, encoder blocks, final normalization, and classifier.

```text
data_loader.py
```

Builds dataset loaders using `torchvision`. The current verification flow uses MNIST.

```text
checkpoints/model_encoder_seed1234.pt
checkpoints/model_encoder_seed2026.pt
```

Fixed model checkpoints used to generate reproducible reference payloads.

## Model Configuration

```text
dataset: MNIST
image_size: 28
patch_size: 4
embedding_dim: 64
encoder_blocks: 6
attention_heads: 4
classes: 10
```

## How This Step Is Used

The export scripts in `../02_fixed_point_export/` import `model.py` and `data_loader.py`, load the checkpoint files, run inference, and save intermediate tensors for hardware verification.

## Result Of This Step

```text
PyTorch model + fixed checkpoints -> trusted floating-point inference reference
```
