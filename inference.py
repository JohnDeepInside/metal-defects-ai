"""
Single-image inference for Metal Surface Defect Detection.

Usage:
    python inference.py --image path/to/image.jpg --checkpoint outputs/checkpoints/best_model.pth
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
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", type=Path, required=True,
                        help="Path to the input image")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save prediction visualization to this path")
    return parser.parse_args()


def predict(
    image_path: Path,
    checkpoint_path: Path,
) -> tuple[str, float, dict[str, float]]:
    """
    Run inference on a single image.

    Returns:
        (predicted_class, confidence, all_probabilities)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", "efficientnet_b0")
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)
    class_names = ckpt.get("class_names", config.CLASS_NAMES)

    # Build model
    model = build_model(arch=arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    # Inference
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

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(12, 5),
                                          gridspec_kw={"width_ratios": [1, 1.2]})

    # Image
    ax_img.imshow(image)
    ax_img.set_title(f"Prediction: {predicted_class}\nConfidence: {confidence:.1%}",
                     fontsize=13, fontweight="bold")
    ax_img.axis("off")

    # Bar chart
    names = list(all_probs.keys())
    values = list(all_probs.values())
    colors = ["#2ecc71" if n == predicted_class else "#95a5a6" for n in names]

    bars = ax_bar.barh(names, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=10)

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

    print(f"\nImage:      {args.image}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nAll probabilities:")
    for name, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        print(f"  {name:20s} {prob:.4f}")

    visualize_prediction(args.image, predicted_class, confidence, all_probs,
                         save_path=args.save)


if __name__ == "__main__":
    main()
