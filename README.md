# AneRBC-ANN-and-DL 
# Anemia Detection Using Deep Learning and Explainable AI

## Project Overview

This project investigates automated anemia detection from Red Blood Cell (RBC) images using Deep Learning and Explainable AI (XAI). Three custom CNN architectures (CNN3, CNN4, CNN5) and three transfer learning models (MobileNetV2, ResNet18, and SqueezeNet) were implemented and evaluated using the AneRBC dataset.

Grad-CAM was applied to provide visual explanations of model predictions and improve interpretability.

---

 # Environment Setup

## Google Colab Setup

This project was developed and tested using Google Colab with GPU acceleration.

Enable GPU:

Runtime → Change Runtime Type → GPU

---

## Install Required Packages

Run the following commands before executing the notebook:

```bash
pip install torch torchvision
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install opencv-python
pip install grad-cam
pip install kagglehub
```

---

## Verify Installation

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

Expected output:

```python
True
```

---

# Dataset Download and Preparation

## Dataset Source

AneRBC Dataset:

https://www.kaggle.com/datasets/jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images

---

## Download Dataset

Using KaggleHub:

```python
import kagglehub

path = kagglehub.dataset_download(
    "jocelyndumlao/anerbc-anemia-diagnosis-using-rbc-images"
)

print(path)
```

---

## Dataset Structure

Only the Original_images folders are used:

```text
AneRBC-I/
 ├── Healthy_individuals/
 │    └── Original_images/
 └── Anemic_individuals/
      └── Original_images/

AneRBC-II/
 ├── Healthy_individuals/
 │    └── Original_images/
 └── Anemic_individuals/
      └── Original_images/
```

---

## Dataset Preparation

Steps performed:

1. Load image paths from Healthy and Anemic folders.
2. Assign labels:

   * Healthy = 0
   * Anemic = 1
3. Resize images to 224×224 pixels.
4. Normalize pixel values.
5. Perform train-validation-test split.

Dataset statistics:

* Healthy Images: 500
* Anemic Images: 500
* Total Images: 1000

Split ratio:

* Training: 70%
* Validation: 15%
* Testing: 15%

---

# Training Custom CNN Models

Three custom CNN architectures were implemented.

## CNN3

Train:

```python
cnn3 = CNN3().to(device)
train_model(cnn3, train_loader, epochs=15)
```

Evaluate:

```python
evaluate_model(cnn3, test_loader)
```

---

## CNN4

Train:

```python
cnn4 = CNN4().to(device)
train_model(cnn4, train_loader, epochs=15)
```

Evaluate:

```python
evaluate_model(cnn4, test_loader)
```

---

## CNN5

Train:

```python
cnn5 = CNN5().to(device)
train_model(cnn5, train_loader, epochs=15)
```

Evaluate:

```python
evaluate_model(cnn5, test_loader)
```

---

# Training Transfer Learning Models

Pretrained ImageNet weights were used.

## MobileNetV2

Train:

```python
mobilenet = MobileNetV2().to(device)
train_model(mobilenet, train_loader, epochs=10)
```

Evaluate:

```python
evaluate_model(mobilenet, test_loader)
```

---

## ResNet18

Train:

```python
resnet = ResNet18().to(device)
train_model(resnet, train_loader, epochs=10)
```

Evaluate:

```python
evaluate_model(resnet, test_loader)
```

---

## SqueezeNet

Train:

```python
squeezenet = SqueezeNet().to(device)
train_model(squeezenet, train_loader, epochs=10)
```

Evaluate:

```python
evaluate_model(squeezenet, test_loader)
```

---

# Model Evaluation

For every model, the following evaluation metrics are calculated:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example:

```python
evaluate_model(model, test_loader)
```

Confusion Matrix:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
```

Visualization:

```python
sns.heatmap(cm, annot=True, fmt="d")
plt.show()
```

---

# Explainable AI (XAI)

Grad-CAM was applied to:

* CNN5 (Best Custom CNN)
* ResNet18 (Best Transfer Learning Model)

---

## Generate Grad-CAM

```python
cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)
```

Generate activation map:

```python
grayscale_cam = cam(
    input_tensor=input_tensor
)[0]
```

Overlay heatmap:

```python
visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)
```

---

# Running the Complete Project

To reproduce all results:

1. Install required packages.
2. Download the AneRBC dataset.
3. Run dataset preparation cells.
4. Train CNN3, CNN4, and CNN5.
5. Train MobileNetV2, ResNet18, and SqueezeNet.
6. Run evaluation cells.
7. Generate confusion matrices.
8. Create model comparison table.
9. Generate Grad-CAM visualizations for CNN5 and ResNet18.
10. Review XAI interpretations and final results.
