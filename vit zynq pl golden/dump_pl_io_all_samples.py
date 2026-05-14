import os
import site
import sys

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import torch

from data_loader import get_loader
from model import VisionTransformer


SEED_MODELS = {
    "seed1234": "model_encoder_seed1234.pt",
    "seed2026": "model_encoder_seed2026.pt",
}

ROOT_DIR = "./pl_io_20_samples"
MAX_SAMPLES = 20

WIDTH = 16
FRAC = 12


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def tensor_to_fixed_int(t: torch.Tensor, frac: int = FRAC, width: int = WIDTH) -> torch.Tensor:
    x = torch.round(t * (1 << frac)).to(torch.int64)
    minv = -(1 << (width - 1))
    maxv = (1 << (width - 1)) - 1
    return torch.clamp(x, minv, maxv)


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


def get_test_loader():
    _, test_loader = get_loader(
        type(
            "Args",
            (),
            {
                "dataset": "mnist",
                "image_size": 28,
                "patch_size": 4,
                "n_channels": 1,
                "data_path": "./data/",
                "batch_size": 1,
                "num_workers": 0,
            },
        )()
    )
    return test_loader


def write_flat_row(f, values):
    f.write(" ".join(str(int(v)) for v in values))
    f.write("\n")


def export_one_seed(seed_name: str, model_path: str):
    print(f"\n========== Exporting first {MAX_SAMPLES} PL IO samples for {seed_name} ==========")
    out_dir = os.path.join(ROOT_DIR, seed_name, "fixed")
    ensure_dir(out_dir)

    model = build_full_scale_encoder_model()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    test_loader = get_test_loader()
    sample_count = 0

    embedding_path = os.path.join(out_dir, "embedding_output_20.txt")
    logits_path = os.path.join(out_dir, "logits_20.txt")
    pred_path = os.path.join(out_dir, "pred_class_20.txt")
    label_path = os.path.join(out_dir, "label_20.txt")

    with open(embedding_path, "w", encoding="utf-8") as f_embedding, \
         open(logits_path, "w", encoding="utf-8") as f_logits, \
         open(pred_path, "w", encoding="utf-8") as f_pred, \
         open(label_path, "w", encoding="utf-8") as f_label:

        with torch.no_grad():
            for x_img, label in test_loader:
                embedding = model.embedding(x_img)
                x = model.encoder(embedding)
                x = model.norm(x)
                logits = model.classifier(x)
                pred_class = torch.argmax(logits, dim=-1).to(torch.int64)

                batch_size = x_img.shape[0]
                for b in range(batch_size):
                    if sample_count >= MAX_SAMPLES:
                        break

                    embedding_fixed = tensor_to_fixed_int(embedding[b]).reshape(-1)
                    logits_fixed = tensor_to_fixed_int(logits[b]).reshape(-1)

                    write_flat_row(f_embedding, embedding_fixed.tolist())
                    write_flat_row(f_logits, logits_fixed.tolist())
                    f_pred.write(f"{int(pred_class[b].item())}\n")
                    f_label.write(f"{int(label[b].item())}\n")

                    sample_count += 1
                    if sample_count % 1000 == 0:
                        print(f"  exported {sample_count} samples")

                if sample_count >= MAX_SAMPLES:
                    break

    with open(os.path.join(out_dir, "sample_count.txt"), "w", encoding="utf-8") as f:
        f.write(f"{sample_count}\n")

    print(f"Done: {seed_name}, samples={sample_count}")
    print(f"  output -> {os.path.abspath(out_dir)}")


def main():
    ensure_dir(ROOT_DIR)
    for seed_name, model_path in SEED_MODELS.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Cannot find {model_path}.")
        export_one_seed(seed_name, model_path)

    print("\nFirst-20 PL IO sample exports done.")
    print(f"Root directory: {os.path.abspath(ROOT_DIR)}")


if __name__ == "__main__":
    main()
