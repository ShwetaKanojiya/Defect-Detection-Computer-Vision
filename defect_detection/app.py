import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import tempfile
from gradcam import generate_gradcam_overlay

# Streamlit Page Configuration
st.set_page_config(
    page_title="Automated Defect Detection", 
    page_icon="⚙️", 
    layout="centered"
)

st.title("Smart Industrial Quality Inspection System")
st.markdown("""
***Quality Control & Defect Detection***
""")

# Cache the model to prevent reloading on every Streamlit interaction
@st.cache_resource
def load_model(model_path="model.h5"):
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

if model is None:
    st.warning("⚠️ **Model file (`model.h5`) not found!**")
    st.info("Please train the model first by running `python train.py` and ensure `model.h5` is in the same directory as this script.")
    st.stop()

# Image Uploader Interface
st.markdown("### 1. Upload Product Image")
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Error handling wrapper
    try:
        # Load User Image
        image = Image.open(uploaded_file)
        
        # We'll use 2 columns to put original beside Grad-CAM result
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Input Provided")
            st.image(image, caption="Uploaded Product Image", use_container_width=True)
            
        # Analysis phase
        with st.spinner("Executing Model Inference..."):
            # Prepare image per MobileNetV2 input requirements
            img_resized = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # Predict
            prediction = model.predict(img_array)
            probability = prediction[0][0]
            
            # Typically, generators map folders alphabetically: 
            # 0 -> 'defective', 1 -> 'ok'. 
            # So a probability > 0.5 implies 'OK', approaching 1.0.
            # A probability < 0.5 implies 'Defective', approaching 0.0.
            if probability > 0.5:
                result_label = "STANDARD (OK)"
                # Distance to 1 indicates confidence for OK class
                confidence = probability * 100 
                # UI Styling for good
                st.success(f"**Prediction Outcome:** {result_label} ✅")
            else:
                result_label = "DEFECT DETECTED"
                # Distance to 0 indicates confidence for DEFECTIVE class
                confidence = (1 - probability) * 100 
                # UI Styling for error
                st.error(f"**Prediction Outcome:** {result_label} ❌")
            
            # Show numerical confidence
            st.metric(label="Model Confidence Score", value=f"{confidence:.2f} %")
            
        with st.spinner("Generating Explainability Maps..."):
            st.markdown("---")
            st.markdown("### 2. Explainability Analysis (Grad-CAM)")
            st.markdown("The heatmap below visualizes the regions our AI focused on to make its determination. Warmer colors signify higher influence.")
            
            # Grad-CAM function expects a file path. Let's dump Streamlit's image to a temp location.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                image.save(tmp_file.name)
                tmp_path = tmp_file.name
                
            # Perform Map Overlay
            gradcam_img = generate_gradcam_overlay(tmp_path, model, size=(224, 224))
            
            with col2:
                st.markdown("#### Analysis Insight")
                if gradcam_img is not None:
                    st.image(gradcam_img, caption="Grad-CAM Focus Overlay", use_container_width=True)
                else:
                    st.warning("Overlay generation skipped. See console logs.")
                    
            # Auto-cleanup after rendering
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        st.error(f"An error occurred over the image parsing process: {str(e)}")
        st.info("Please ensure your uploaded image format is not corrupted.")
