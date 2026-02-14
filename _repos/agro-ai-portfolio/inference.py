"""
Single-image inference for Plant Disease Detection.

Usage:
    python inference.py --image path/to/leaf.jpg --checkpoint outputs/checkpoints/best_model.pth
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
    parser = argparse.ArgumentParser(description="Classify a plant leaf image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Show top-K predictions")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save prediction visualization to this path")
    return parser.parse_args()


def predict(
    image_path: Path,
    checkpoint_path: Path,
    top_k: int = 5,
) -> tuple[str, float, list[tuple[str, float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    arch = ckpt.get("arch", "resnet18")
    num_classes = ckpt.get("num_classes", config.NUM_CLASSES)
    class_names = ckpt.get("class_names", None)

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

    top_probs, top_indices = probs.topk(top_k)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    predicted_class = class_names[top_indices[0].item()]
    confidence = top_probs[0].item()
    top_predictions = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    return predicted_class, confidence, top_predictions


def visualize_prediction(
    image_path: Path,
    predicted_class: str,
    confidence: float,
    top_predictions: list[tuple[str, float]],
    save_path: Path | None = None,
) -> None:
    image = Image.open(image_path).convert("RGB")

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(14, 5),
                                          gridspec_kw={"width_ratios": [1, 1.3]})

    ax_img.imshow(image)
    label = predicted_class.replace("___", " — ").replace("_", " ")
    ax_img.set_title(f"{label}\nConfidence: {confidence:.1%}",
                     fontsize=12, fontweight="bold")
    ax_img.axis("off")

    names = [p[0].replace("___", " — ").replace("_", " ") for p in top_predictions]
    values = [p[1] for p in top_predictions]
    colors = ["#2ecc71" if i == 0 else "#95a5a6" for i in range(len(names))]

    bars = ax_bar.barh(names[::-1], values[::-1], color=colors[::-1],
                       edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values[::-1]):
        ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=10)

    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title("Top-5 Predictions", fontweight="bold")

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

    predicted_class, confidence, top_predictions = predict(
        args.image, args.checkpoint, top_k=args.top_k,
    )

    label = predicted_class.replace("___", " — ").replace("_", " ")
    print(f"\nImage:      {args.image}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nTop-{args.top_k} predictions:")
    for name, prob in top_predictions:
        print(f"  {name:40s} {prob:.4f}")

    visualize_prediction(args.image, predicted_class, confidence,
                         top_predictions, save_path=args.save)


if __name__ == "__main__":
    main()
