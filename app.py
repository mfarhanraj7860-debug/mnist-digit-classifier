import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.title("MNIST Digit Classifier")

def load_model():
    return tf.keras.models.load_model("mnist_model.keras")

st.write("Upload a 28x28 grayscale digit image.")

uploaded = st.file_uploader("Choose image", type=["png","jpg","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L").resize((28,28))
    st.image(img, caption="Input Image", width=150)

    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 784)

    model = load_model()
    pred = model.predict(arr, verbose=0)
    digit = int(np.argmax(pred))

    st.success(f"Predicted Digit: {digit}")
