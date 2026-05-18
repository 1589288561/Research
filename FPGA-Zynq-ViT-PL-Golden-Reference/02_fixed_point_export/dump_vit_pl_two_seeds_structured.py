import math
import os
from pathlib import Path
import site
import sys

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_REF_DIR = PROJECT_ROOT / "01_python_reference"
if str(PYTHON_REF_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_REF_DIR))

import torch

from data_loader import get_loader
from model import VisionTransformer


SEED_MODELS = {
    "seed1234": PROJECT_ROOT / "01_python_reference" / "checkpoints" / "model_encoder_seed1234.pt",
    "seed2026": PROJECT_ROOT / "01_python_reference" / "checkpoints" / "model_encoder_seed2026.pt",
}

ROOT_DIR = Path(__file__).resolve().parent / "exports"

WIDTH = 16
FRAC = 12


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_fixed_int(t: torch.Tensor, frac: int = FRAC, width: int = WIDTH) -> torch.Tensor:
    x = torch.round(t * (1 << frac)).to(torch.int64)
    minv = -(1 << (width - 1))
    maxv = (1 << (width - 1)) - 1
    return torch.clamp(x, minv, maxv)


def save_txt_float(t: torch.Tensor, path: str):
    t = t.detach().cpu()
    with open(path, "w", encoding="utf-8") as f:
        if t.ndim == 0:
            f.write(f"{t.item()}\n")
        elif t.ndim == 1:
            for v in t.tolist():
                f.write(f"{v}\n")
        elif t.ndim == 2:
            for row in t.tolist():
                f.write(" ".join(str(v) for v in row) + "\n")
        else:
            raise ValueError(f"Unsupported ndim for float txt export: {t.ndim}")


def save_txt_fixed(t: torch.Tensor, path: str):
    x = tensor_to_fixed_int(t).detach().cpu()
    with open(path, "w", encoding="utf-8") as f:
        if x.ndim == 0:
            f.write(f"{int(x.item())}\n")
        elif x.ndim == 1:
            f.write(" ".join(str(int(v)) for v in x.tolist()) + "\n")
        elif x.ndim == 2:
            for row in x.tolist():
                f.write(" ".join(str(int(v)) for v in row) + "\n")
        else:
            raise ValueError(f"Unsupported ndim for fixed txt export: {x.ndim}")


def save_tensor_both(t: torch.Tensor, base_dir: str, name: str):
    float_dir = os.path.join(base_dir, "float")
    fixed_dir = os.path.join(base_dir, "fixed")
    ensure_dir(float_dir)
    ensure_dir(fixed_dir)
    torch.save(t.detach().cpu(), os.path.join(float_dir, f"{name}.pt"))
    save_txt_float(t, os.path.join(float_dir, f"{name}.txt"))
    save_txt_fixed(t, os.path.join(fixed_dir, f"{name}.txt"))


def save_scalar_int_both(v: int, base_dir: str, name: str):
    float_dir = os.path.join(base_dir, "float")
    fixed_dir = os.path.join(base_dir, "fixed")
    ensure_dir(float_dir)
    ensure_dir(fixed_dir)
    for out_dir in [float_dir, fixed_dir]:
        with open(os.path.join(out_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write(f"{int(v)}\n")


def save_linear_params(layer: torch.nn.Linear, base_dir: str, prefix: str):
    save_tensor_both(layer.weight.detach().cpu(), base_dir, f"{prefix}_weight")
    save_tensor_both(layer.bias.detach().cpu(), base_dir, f"{prefix}_bias")


def save_vec_params(v: torch.Tensor, base_dir: str, prefix: str):
    save_tensor_both(v.detach().cpu(), base_dir, prefix)


def build_full_scale_encoder_model():
    return VisionTransformer(
        n_channels=1,
        embed_dim=64,
        n_layers=6,
        n_attention_heads=4,
        forward_mul=2,
        image_size=28,
        patch_size=4,
        n_classes=10,
    )


def get_one_test_sample():
    _, test_loader = get_loader(
        type(
            "Args",
            (),
            {
                "dataset": "mnist",
                "image_size": 28,
                "patch_size": 4,
                "n_channels": 1,
                "data_path": str(PROJECT_ROOT / "data"),
                "batch_size": 1,
                "num_workers": 0,
            },
        )()
    )
    return next(iter(test_loader))


def export_block(seed_root: str, block_idx: int, block, block_input: torch.Tensor) -> torch.Tensor:
    block_name = f"block{block_idx + 1}"
    basic_dir = os.path.join(seed_root, block_name, "basic")
    debug_dir = os.path.join(seed_root, block_name, "debug")

    attn = block.attention

    norm1_output = block.norm1(block_input)
    q_output = attn.wq(norm1_output)
    k_output = attn.wk(norm1_output)
    v_output = attn.wv(norm1_output)

    bsz, tokens, dim = q_output.shape
    heads = attn.num_heads
    head_dim = attn.head_dim

    qh = q_output.view(bsz, tokens, heads, head_dim).transpose(1, 2)
    kh = k_output.view(bsz, tokens, heads, head_dim).transpose(1, 2)
    vh = v_output.view(bsz, tokens, heads, head_dim).transpose(1, 2)

    score_output = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(head_dim)
    prob_output = torch.softmax(score_output, dim=-1)
    context_h = torch.matmul(prob_output, vh)
    attention_only_output = context_h.transpose(1, 2).reshape(bsz, tokens, dim)

    att_block_output = block_input + attention_only_output
    norm2_output = block.norm2(att_block_output)
    fc1_output = block.fc1(norm2_output)
    gelu_output = block.activation(fc1_output)
    fc2_output = block.fc2(gelu_output)
    block_output = att_block_output + fc2_output

    save_tensor_both(block_input[0], basic_dir, "block_input")
    save_tensor_both(block_output[0], basic_dir, "block_output")

    save_vec_params(block.norm1.weight, basic_dir, "norm1_weight")
    save_vec_params(block.norm1.bias, basic_dir, "norm1_bias")
    save_vec_params(block.norm2.weight, basic_dir, "norm2_weight")
    save_vec_params(block.norm2.bias, basic_dir, "norm2_bias")

    save_linear_params(attn.wq, basic_dir, "wq")
    save_linear_params(attn.wk, basic_dir, "wk")
    save_linear_params(attn.wv, basic_dir, "wv")
    save_linear_params(block.fc1, basic_dir, "fc1")
    save_linear_params(block.fc2, basic_dir, "fc2")

    save_tensor_both(norm1_output[0], debug_dir, "norm1_output")
    save_tensor_both(q_output[0], debug_dir, "q_output")
    save_tensor_both(k_output[0], debug_dir, "k_output")
    save_tensor_both(v_output[0], debug_dir, "v_output")
    for head_idx in range(score_output.shape[1]):
        save_tensor_both(score_output[0, head_idx], debug_dir, f"score_output_head{head_idx}")
        save_tensor_both(prob_output[0, head_idx], debug_dir, f"prob_output_head{head_idx}")
    save_tensor_both(attention_only_output[0], debug_dir, "attention_only_output")
    save_tensor_both(att_block_output[0], debug_dir, "att_block_output")
    save_tensor_both(norm2_output[0], debug_dir, "norm2_output")
    save_tensor_both(fc1_output[0], debug_dir, "fc1_output")
    save_tensor_both(gelu_output[0], debug_dir, "gelu_output")
    save_tensor_both(fc2_output[0], debug_dir, "fc2_output")
    save_tensor_both(block_output[0], debug_dir, "mlp_block_output")

    return block_output


def export_norm(seed_root: str, norm_name: str, norm, x: torch.Tensor) -> torch.Tensor:
    base_dir = os.path.join(seed_root, norm_name)
    y = norm(x)
    save_tensor_both(x[0], base_dir, "norm_input")
    save_tensor_both(y[0], base_dir, "norm_output")
    save_vec_params(norm.weight, base_dir, "norm_weight")
    save_vec_params(norm.bias, base_dir, "norm_bias")
    return y


def export_classifier(seed_root: str, classifier, x: torch.Tensor):
    base_dir = os.path.join(seed_root, "classifier")
    cls_input = x[:, 0, :]
    fc1_output = classifier.fc1(cls_input)
    tanh_output = classifier.activation(fc1_output)
    logits = classifier.fc2(tanh_output)
    pred_class = torch.argmax(logits, dim=-1).to(torch.int64)

    save_tensor_both(cls_input[0], base_dir, "cls_input")
    save_tensor_both(fc1_output[0], base_dir, "classifier_fc1_output")
    save_tensor_both(tanh_output[0], base_dir, "classifier_tanh_output")
    save_tensor_both(logits[0], base_dir, "logits")
    save_scalar_int_both(int(pred_class[0].item()), base_dir, "pred_class")

    save_linear_params(classifier.fc1, base_dir, "classifier_fc1")
    save_linear_params(classifier.fc2, base_dir, "classifier_fc2")


def export_one_seed(seed_name: str, model_path: str):
    print(f"\n========== Exporting {seed_name} from {model_path} ==========")
    seed_root = os.path.join(ROOT_DIR, seed_name)
    ensure_dir(seed_root)

    model = build_full_scale_encoder_model()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    x_img, y = get_one_test_sample()

    with torch.no_grad():
        x = model.embedding(x_img)
        save_tensor_both(x_img[0, 0], os.path.join(seed_root, "input_debug"), "image")
        save_tensor_both(y, os.path.join(seed_root, "input_debug"), "label")
        save_tensor_both(x[0], os.path.join(seed_root, "embedding_output"), "embedding_output")

        for block_idx in range(6):
            x = export_block(seed_root, block_idx, model.encoder[block_idx], x)

        x = export_norm(seed_root, "encoder_final_norm", model.encoder[6], x)
        x = export_norm(seed_root, "model_norm", model.norm, x)
        save_tensor_both(x[0], os.path.join(seed_root, "pre_classifier"), "pre_classifier")
        export_classifier(seed_root, model.classifier, x)

    print(f"Done: {seed_name}")
    print(f"  root -> {os.path.abspath(seed_root)}")


def main():
    ensure_dir(ROOT_DIR)
    for seed_name, model_path in SEED_MODELS.items():
        if not model_path.exists():
            raise FileNotFoundError(
                f"Cannot find {model_path}. Put the checkpoint under 01_python_reference/checkpoints."
            )
        export_one_seed(seed_name, model_path)

    print("\nAll PS-to-PL exports done.")
    print(f"Root directory: {os.path.abspath(ROOT_DIR)}")


if __name__ == "__main__":
    main()
