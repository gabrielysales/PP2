import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from utils.preprocess import preprocess_image

st.set_page_config(page_title="Classificador de Pneumonia", layout="centered")

st.write("STREAMLIT ESTÁ RODANDO")  # TESTE

@st.cache_resource
def load_my_model():
    model = load_model("modelo_pneumonia.h5")
    return model

model = load_my_model()

class_names = ["Normal", "Pneumonia"]

st.title("🩺 Classificador de Pneumonia por Raio-X")
st.write("Modelo treinado no Kaggle (Chest X-Ray Pneumonia) e desenvolvido no Google Colab.")

uploaded_file = st.file_uploader("Envie uma imagem de raio-X (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Imagem enviada", width=300)

    input_tensor = preprocess_image(img)

    prob = model.predict(input_tensor)[0][0]

    class_index = int(prob > 0.5)
    class_name = class_names[class_index]

    st.subheader("🔍 Resultado do modelo:")
    st.write(f"**Classe:** {class_name}")
    st.write(f"**Probabilidade de Pneumonia:** {prob:.4f}")

    if class_name == "Pneumonia":
        st.error("⚠️ Possível caso de PNEUMONIA.")
    else:
        st.success("🟢 Parece NORMAL.")
