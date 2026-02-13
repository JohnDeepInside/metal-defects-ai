Metal Surface Defect Detector
A deep learning model that identifies defects on steel surfaces from a single photo. Trained on 1,656 images across 6 defect types.
Final accuracy: 100% on validation set.
Why this matters
Manufacturing lines lose time and money when defective parts slip through quality control. This model automates visual inspection — no human eye needed, no missed defects.
What it detects
6 types of steel surface defects: Crazing
Inclusion Patches
Pitted surface Rolled-in scale Scratches
How it was built
Dataset: NEU Metal Surface Defects — 1,656 training images, 72 validation images Model: EfficientNet-B0 pretrained on ImageNet, fine-tuned for defect detection Framework: PyTorch
Training: 15 epochs on GPU (Google Colab T4)
    Input size: 224×224 px
  Results
   Epoch
1 2 5 15
Project structure
  metal-defects-ai/
  ├── train.py
  └── README.md
Run it yourself
Train Acc
93.2% 98.3% 99.0% 99.7%
# Training script
Val Acc
100% 100% 100% 100%
                                 # Install dependencies
  pip install torch torchvision
  # Train from scratch
  python train.py
Tech stack
Python 3.12
PyTorch
torchvision (EfficientNet-B0) Google Colab (T4 GPU)
  
Author
Built by Nikolai Shatikhin as part of an AI/ML portfolio. Open to freelance projects in computer vision and image classification.
Reach out via GitHub issues or direct message.
