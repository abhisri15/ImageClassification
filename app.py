import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model("saved_model")
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Set page title and favicon
st.set_page_config(
    page_title="Image Prediction with Streamlit",
    page_icon="🌸"
)

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
        body {
            color: #1E1E1E;
            background-color: #F5F5F5;
        }
        .st-bc {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and description
st.title("🌺 Image Prediction with Streamlit")
st.write("Upload an image and let the model predict the flower!")

# Upload image through Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Preprocess and predict the image
    image = Image.open(uploaded_image).resize((180, 180))
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Display prediction results as bars
    st.markdown("<h2>Prediction Results:</h2>", unsafe_allow_html=True)
    result_data = {class_name: float(prob) for class_name, prob in zip(class_names, score)}
    st.bar_chart(result_data, width=400, height=400)

    # Display detailed probabilities
    st.markdown("<h3>Detailed Probabilities:</h3>", unsafe_allow_html=True)
    for class_name, prob in zip(class_names, score):
        st.write(f"{class_name}: {float(prob):.4f}")
