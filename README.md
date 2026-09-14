# Chest X-ray Pneumonia Detection with Transfer Learning

[![Live App](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?logo=streamlit&logoColor=white)](STREAMLIT_URL)

A deep-learning project for binary pneumonia classification from chest X-ray images using **MobileNetV2** as a frozen ImageNet feature extractor and a custom classification head.

## Streamlit Application

Upload a PNG or JPEG chest X-ray, run the MobileNetV2 model, and view the predicted class, confidence, and pneumonia probability. The app applies the same `preprocess_input` transformation and 160 × 160 RGB input size used during model training.

## Results

- Test accuracy: **80.8%**
- Sensitivity / recall: **98.2%**
- Specificity: **51.7%**
- ROC-AUC: **0.9539**
- Validation-selected threshold: **0.50**
- Held-out test set: **624 images**

The model prioritizes sensitivity, but its 51.7% specificity means false-positive pneumonia alerts are possible.

## Technologies

Python, TensorFlow, Keras, MobileNetV2, NumPy, Pillow, Streamlit, scikit-learn, Matplotlib, Pandas, and Jupyter.

## Project Structure

```text
.
├── app.py
├── pneumonia_mobilenetv2.keras
├── chest_xray_pneumonia_detection.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## How to Run

```bash
git clone https://github.com/Karim797/chest-xray-pneumonia-transfer-learning.git
cd chest-xray-pneumonia-transfer-learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Training Workflow

The notebook covers dataset download, stratified train/validation splitting, MobileNetV2 feature extraction, class-weighted head training, threshold selection, held-out evaluation, error analysis, and model persistence.

Dataset: Kaggle `paultimothymooney/chest-xray-pneumonia`. The dataset is not stored in this repository.

## Responsible Use

This application is an educational portfolio demonstration. It is **not a medical device**, does not provide a clinical diagnosis, and must not replace review by a qualified healthcare professional.

## License

Released under the MIT License.
