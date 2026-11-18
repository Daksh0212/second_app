from pathlib import Path
from io import BytesIO

import streamlit as st
from PIL import Image
from fastai.vision.all import load_learner, PILImage

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"

st.set_page_config(
    page_title="Dog or Cat Classifier",
    page_icon="🐾",
    layout="centered",
)

st.title("🐾 Dog or Cat Classifier")
st.write(
    "Upload an image and let your Fast.ai model classify it. "
    "The app displays the predicted class and confidence scores."
)


@st.cache_resource
def load_model(model_path: str):
    # cpu=True lets the app run without a GPU.
    return load_learner(model_path, cpu=True)


if not MODEL_PATH.exists():
    st.error("model.pkl was not found in the app's repository folder.")
    st.info(
        "After training in your notebook, run "
        "`learn.export('model.pkl')`, download the file, and upload it "
        "beside app.py."
    )
    st.stop()

try:
    learn = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The model could not be loaded. Check that model.pkl is a "
             "Fast.ai exported learner and that the dependencies installed.")
    st.exception(exc)
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)

if uploaded_file is not None:
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Classify image", type="primary", use_container_width=True):
            with st.spinner("Classifying..."):
                fastai_image = PILImage.create(image)
                prediction, prediction_index, probabilities = learn.predict(
                    fastai_image
                )

            predicted_label = str(prediction)
            confidence = float(probabilities[int(prediction_index)])

            st.success(f"Prediction: **{predicted_label}**")
            st.metric("Confidence", f"{confidence:.1%}")

            st.subheader("Class probabilities")
            class_names = [str(label) for label in learn.dls.vocab]
            probability_values = [float(value) for value in probabilities]
            probability_data = {
                name: value
                for name, value in zip(class_names, probability_values)
            }
            st.bar_chart(probability_data)

            st.caption(
                "Probabilities are model estimates, not guarantees. "
                "Results depend on the training data and image quality."
            )
    except Exception as exc:
        st.error("The image could not be processed.")
        st.exception(exc)
else:
    st.caption("Supported formats: JPG, JPEG, PNG, WEBP and BMP.")
