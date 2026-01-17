import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------
# Page Config
# --------------------
st.set_page_config(
    page_title="Skin Disease AI Detector",
    page_icon="🧬",
    layout="wide"
)

# --------------------
# Classes
# --------------------
CLASSES = ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]

CLASS_NAMES = {
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "BCC": "Basal Cell Carcinoma",
    "AKIEC": "Actinic Keratosis",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular Lesion"
}

CAUSES = {
    "AKIEC": "Caused by long-term sun (UV) exposure damaging skin cells.",
    "BCC": "Caused by UV radiation leading to abnormal growth of basal skin cells.",
    "BKL": "Caused by aging and sun exposure leading to benign skin growth.",
    "DF": "Caused by minor skin injury or insect bite triggering fibrous tissue growth.",
    "MEL": "Caused by DNA damage from UV light leading to uncontrolled pigment cell growth.",
    "NV": "Caused by clustering of melanocytes (pigment cells).",
    "VASC": "Caused by abnormal blood vessel formation in the skin."
}

PRECAUTIONS = {
    "AKIEC": "Use sunscreen, avoid midday sun, and monitor skin changes.",
    "BCC": "Avoid sun exposure, use SPF 50+, and get regular skin checkups.",
    "BKL": "Protect skin from sun and avoid picking or scratching lesions.",
    "DF": "Avoid skin trauma and consult a doctor if it changes.",
    "MEL": "Seek immediate medical attention, avoid sun, and get a biopsy.",
    "NV": "Monitor moles for size, color, or shape changes.",
    "VASC": "Avoid injury and consult a dermatologist if bleeding occurs."
}

# --------------------
# Load model
# --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_mobilenetv2.h5")

model = load_model()

# --------------------
# UI
# --------------------
st.title("🧬 Skin Disease Detection System")
st.write("Upload a skin lesion image to detect disease, cause, and precautions.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1.2,1])

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224,224))
    img = np.array(img).astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = np.argmax(preds)
    cls = CLASSES[idx]
    conf = preds[idx] * 100

    with col2:
        st.subheader("Prediction")
        st.metric("Disease", CLASS_NAMES[cls])
        st.metric("Confidence", f"{conf:.2f}%")
        st.progress(float(conf/100))

        st.markdown("### Cause")
        st.write(CAUSES[cls])

        st.markdown("### Precaution")
        st.write(PRECAUTIONS[cls])

    st.markdown("---")
    st.subheader("Class Probabilities")

    for i in np.argsort(preds)[::-1]:
        st.write(f"{CLASS_NAMES[CLASSES[i]]}: {preds[i]*100:.2f}%")
        st.progress(float(preds[i]))

else:
    st.info("Upload a skin image to begin.")
