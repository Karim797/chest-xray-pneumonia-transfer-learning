# Chest X-ray Pneumonia Detection with Transfer Learning

A deep-learning project for binary pneumonia classification from chest X-ray images using **MobileNetV2** as a frozen ImageNet feature extractor and a custom classification head.

## Project Overview

This notebook covers the complete workflow:

- Downloading the Chest X-Ray Pneumonia dataset from Kaggle
- Creating stratified train/validation splits
- Extracting image features with MobileNetV2
- Training a custom classification head
- Handling class imbalance with class weights
- Selecting the classification threshold using validation data
- Evaluating with accuracy, sensitivity, specificity, F1 score, confusion matrix, and ROC-AUC
- Visualizing correct predictions and failure cases
- Saving the trained model and evaluation results

## Results

Results stored in the current notebook run:

- **Test accuracy:** 80.8%
- **Sensitivity / Recall:** 98.2%
- **Specificity:** 51.7%
- **ROC-AUC:** 0.95
- **Held-out test set:** 624 images
- **Majority-class baseline accuracy:** 62.5%

The notebook selects the decision threshold using the validation set rather than tuning it on the test set.

## Model

- Backbone: MobileNetV2
- Pretraining: ImageNet
- Input size: 160 × 160 RGB
- Transfer-learning strategy: frozen backbone + trainable classification head
- Framework: TensorFlow / Keras

## Dataset

The notebook downloads the Kaggle **Chest X-Ray Images (Pneumonia)** dataset:

`paultimothymooney/chest-xray-pneumonia`

The dataset itself is intentionally not included in this repository.

## How to Run

1. Clone the repository.
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Set your Kaggle credentials in the notebook or through environment variables.
4. Open `chest_xray_pneumonia_detection.ipynb`.
5. Run the cells in order.

## Repository Structure

```text
chest-xray-pneumonia-transfer-learning/
├── chest_xray_pneumonia_detection.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Notes

The extracted dataset, cached features, generated model files, and other large artifacts are excluded from Git tracking.

## Disclaimer

This project is for educational and portfolio purposes only. It is not intended for clinical diagnosis or medical decision-making.
