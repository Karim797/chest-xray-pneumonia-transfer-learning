from pathlib import Path
import json
import tempfile
import zipfile

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


MODEL_PATH = Path(__file__).with_name("pneumonia_mobilenetv2.keras")
IMAGE_SIZE = (160, 160)
PNEUMONIA_THRESHOLD = 0.50

st.set_page_config(
    page_title="Chest X-ray Pneumonia Detection",
    page_icon="🩻",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading MobileNetV2 model...")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")

    # A .keras file is a ZIP archive containing the architecture and HDF5
    # weights. Loading its two components separately avoids an OSError raised
    # by Keras' archive loader on some Streamlit Community Cloud runtimes.
    extraction_dir = Path(tempfile.mkdtemp(prefix="pneumonia_model_"))
    weights_path = extraction_dir / "model.weights.h5"

    with zipfile.ZipFile(str(MODEL_PATH), "r") as archive:
        required = {"config.json", "model.weights.h5"}
        missing = required.difference(archive.namelist())
        if missing:
            raise ValueError(f"Invalid model archive; missing: {', '.join(sorted(missing))}")

        model_config = archive.read("config.json").decode("utf-8")
        with archive.open("model.weights.h5") as source, weights_path.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)

    model = tf.keras.models.model_from_json(model_config)
    model.load_weights(str(weights_path))
    return model


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    display_image = image.copy()
    resized = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    array = preprocess_input(array)
    return display_image, np.expand_dims(array, axis=0)


st.title("Chest X-ray Pneumonia Detection")
st.caption("MobileNetV2 transfer-learning classifier for NORMAL vs PNEUMONIA chest X-rays.")
st.warning(
    "Educational demonstration only. This model is not a medical diagnosis and must not "
    "replace assessment by a qualified clinician."
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg"],
    help="Use a frontal chest radiograph. Non-X-ray images are outside the model's training distribution.",
)

if uploaded_file is None:
    st.info("Upload a PNG or JPEG chest X-ray to run a prediction.")
    st.stop()

try:
    source_image = Image.open(uploaded_file)
    display_image, model_input = prepare_image(source_image)
except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()

st.image(display_image, caption="Uploaded chest X-ray", use_container_width=True)

if st.button("Analyze X-ray", type="primary", use_container_width=True):
    try:
        with st.spinner("Analyzing image..."):
            model = load_model()
            probabilities = np.asarray(model(model_input, training=False))[0]
    except Exception as exc:
        st.error(
            "The model could not complete the prediction. "
            f"Technical details: {type(exc).__name__}: {exc}"
        )
        st.stop()

    if probabilities.shape != (2,) or not np.all(np.isfinite(probabilities)):
        st.error("The model returned an unexpected output.")
        st.stop()

    normal_probability = float(probabilities[0])
    pneumonia_probability = float(probabilities[1])
    predicted_class = (
        "PNEUMONIA" if pneumonia_probability >= PNEUMONIA_THRESHOLD else "NORMAL"
    )
    confidence = (
        pneumonia_probability if predicted_class == "PNEUMONIA" else normal_probability
    )

    if predicted_class == "PNEUMONIA":
        st.error("Model prediction: PNEUMONIA")
    else:
        st.success("Model prediction: NORMAL")

    col1, col2 = st.columns(2)
    col1.metric("Prediction confidence", f"{confidence:.1%}")
    col2.metric("Pneumonia probability", f"{pneumonia_probability:.1%}")

    st.progress(pneumonia_probability, text="Pneumonia probability")
    st.caption(
        "Decision threshold: 0.50, selected on the validation set. The model's held-out "
        "test sensitivity was 98.2%, but specificity was 51.7%, so false alarms are possible."
    )

with st.expander("Model and evaluation details"):
    st.markdown(
        """
        - Architecture: MobileNetV2 with a custom classification head
        - Input: 160 × 160 RGB image
        - Classes: NORMAL and PNEUMONIA
        - Test accuracy: 80.8%
        - Test sensitivity: 98.2%
        - Test specificity: 51.7%
        - Test ROC-AUC: 0.9539
        """
    )
