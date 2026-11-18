
from pathlib import Path
from io import BytesIO
import traceback

import streamlit as st
from PIL import Image

import torch
import fastai
import plum

from fastai.learner import load_learner
from fastai.vision.all import PILImage


# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"

st.set_page_config(
    page_title="Dog or Cat Classifier",
    page_icon="🐾",
    layout="centered",
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🐾 Dog or Cat Classifier")

st.write(
    "Upload an image and let the Fast.ai model classify it. "
    "The app displays the predicted class and confidence scores."
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):

    # Display useful environment information in the logs
    print("FastAI version:", fastai.__version__)
    print("PyTorch version:", torch.__version__)
    print("Plum version:", getattr(plum, "__version__", "unknown"))

    # Load the exported Fast.ai learner
    learner = load_learner(model_path, cpu=True)

    return learner


if not MODEL_PATH.exists():
    st.error("model.pkl was not found!")

    st.info(
        "Make sure model.pkl is in the same repository folder as app.py."
    )

    st.stop()


try:
    with st.spinner("Loading your AI model..."):
        learn = load_model(str(MODEL_PATH))

    st.success("Model loaded successfully!")

except Exception:
    st.error("The model could not be loaded.")

    st.code(traceback.format_exc())

    st.stop()


# --------------------------------------------------
# ENVIRONMENT INFORMATION
# --------------------------------------------------

with st.expander("Technical information"):
    st.write("FastAI:", fastai.__version__)
    st.write("PyTorch:", torch.__version__)
    st.write(
        "Plum:",
        getattr(plum, "__version__", "version unavailable")
    )
    st.write("Model path:", str(MODEL_PATH))
    st.write("Classes:", [str(x) for x in learn.dls.vocab])


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(
            BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button(
            "Classify image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner("Classifying image..."):

                fastai_image = PILImage.create(image)

                prediction, prediction_index, probabilities = learn.predict(
                    fastai_image
                )

            predicted_label = str(prediction)

            confidence = float(
                probabilities[int(prediction_index)]
            )

            # Prediction result
            st.success(
                f"Prediction: {predicted_label}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )

            # Probability chart
            st.subheader("Class probabilities")

            class_names = [
                str(label) for label in learn.dls.vocab
            ]

            probability_values = [
                float(value) for value in probabilities
            ]

            probability_data = {
                name: value
                for name, value in zip(
                    class_names,
                    probability_values
                )
            }

            st.bar_chart(probability_data)

            st.caption(
                "Probabilities are model estimates, not guarantees. "
                "Results depend on training data and image quality."
            )

    except Exception:
        st.error("Something went wrong while processing the image.")

        st.code(traceback.format_exc())

else:
    st.caption(
        "Supported formats: JPG, JPEG, PNG, WEBP and BMP."
    )
