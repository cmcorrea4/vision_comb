import base64
import platform

import numpy as np
import streamlit as st
from PIL import Image


@st.cache_resource
def cargar_modelo_tm():
    from keras.models import load_model  # import diferido: TF solo se carga en esta página
    return load_model("keras_model.h5", compile=False)


st.title("Reconocimiento de Imágenes")
st.caption(f"Versión de Python: {platform.python_version()}")

with open("OIG5.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
st.html(f'<img src="data:image/jpeg;base64,{b64}" width="350">')

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")

with st.spinner("Cargando modelo Teachable Machine..."):
    model = cargar_modelo_tm()

img_file_buffer = st.camera_input("Toma una Foto", key="cam_tm")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    img = Image.open(img_file_buffer).convert("RGB").resize((224, 224))
    img_array = np.asarray(img)

    # Normalización usada por Teachable Machine
    data[0] = (img_array.astype(np.float32) / 127.0) - 1

    prediction = model.predict(data, verbose=0)

    if prediction[0][0] > 0.5:
        st.header("Izquierda, con Probabilidad: " + str(prediction[0][0]))
    if prediction[0][1] > 0.5:
        st.header("Arriba, con Probabilidad: " + str(prediction[0][1]))
    # if prediction[0][2] > 0.5:
    #     st.header("Derecha, con Probabilidad: " + str(prediction[0][2]))
