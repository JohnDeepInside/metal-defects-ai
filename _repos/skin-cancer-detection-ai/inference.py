"""
Single-image inference for Skin Cancer Detection.

Usage:
    python inference.py --image path/to/lesion.jpg --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

import config
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify a skin lesion image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save", type=Path, default=None,
                        help="Save prediction visualization to this path")
    return parser.parse_args()


def predict(
    image_path: Path,
    checkpoint_path: Path,
) -> tuple[str, float, dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", "mobilenet_v2")
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)
    class_names = ckpt.get("class_names", config.CLASS_NAMES)

    model = build_model(arch=arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).squeeze()

    pred_idx = probs.argmax().item()
    predicted_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()
    all_probs = {name: probs[i].item() for i, name in enumerate(class_names)}

    return predicted_class, confidence, all_probs


def visualize_prediction(
    image_path: Path,
    predicted_class: str,
    confidence: float,
    all_probs: dict[str, float],
    save_path: Path | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    desc = config.CLASS_DESCRIPTIONS.get(predicted_class, predicted_class)

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(13, 5),
                                          gridspec_kw={"width_ratios": [1, 1.3]})

    ax_img.imshow(image)
    ax_img.set_title(f"{desc}\nConfidence: {confidence:.1%}",
                     fontsize=12, fontweight="bold")
    ax_img.axis("off")

    names = [config.CLASS_DESCRIPTIONS.get(n, n) for n in all_probs.keys()]
    values = list(all_probs.values())
    colors = ["#e74c3c" if n == predicted_class else "#95a5a6" for n in all_probs.keys()]

    bars = ax_bar.barh(names, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=9)

    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title("Class Probabilities", fontweight="bold")
    ax_bar.invert_yaxis()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    predicted_class, confidence, all_probs = predict(args.image, args.checkpoint)
    desc = config.CLASS_DESCRIPTIONS.get(predicted_class, predicted_class)

    print(f"\nImage:      {args.image}")
    print(f"Prediction: {predicted_class} — {desc}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        print(f"  {name:6s} ({config.CLASS_DESCRIPTIONS.get(name, ''):30s}) {prob:.4f}")

    # Medical disclaimer
    if predicted_class == "mel":
        print("\n⚠  WARNING: Melanoma detected. Consult a dermatologist immediately.")
    elif predicted_class in ("bcc", "akiec"):
        print("\n⚠  NOTICE: Potentially concerning lesion. Professional evaluation recommended.")

    visualize_prediction(args.image, predicted_class, confidence, all_probs,
                         save_path=args.save)


if __name__ == "__main__":
    main()
