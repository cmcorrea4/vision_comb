import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


@st.cache_resource
def cargar_modelo_yolo():
    try:
        from ultralytics import YOLO  # import diferido: torch solo se carga en esta página
        return YOLO("yolov5su.pt")
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None


st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.")

with st.spinner("Cargando modelo YOLOv5..."):
    model = cargar_modelo_yolo()

if not model:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

with st.sidebar:
    st.title("Parámetros")
    st.subheader("Configuración de detección")
    conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
    max_det = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

picture = st.camera_input("Capturar imagen", key="cam_yolo")

if picture:
    pil_img = Image.open(io.BytesIO(picture.getvalue())).convert("RGB")
    np_img = np.array(pil_img)[..., ::-1]  # RGB → BGR para YOLO

    with st.spinner("Detectando objetos..."):
        try:
            results = model(np_img, conf=conf_threshold, iou=iou_threshold, max_det=int(max_det))
        except Exception as e:
            st.error(f"Error durante la detección: {str(e)}")
            st.stop()

    result = results[0]
    boxes = result.boxes
    annotated_rgb = result.plot()[:, :, ::-1]  # BGR → RGB

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen con detecciones")
        st.image(annotated_rgb, use_container_width=True)

    with col2:
        st.subheader("Objetos detectados")
        if boxes is not None and len(boxes) > 0:
            label_names = model.names
            category_count, category_conf = {}, {}
            for box in boxes:
                cat = int(box.cls.item())
                conf = float(box.conf.item())
                category_count[cat] = category_count.get(cat, 0) + 1
                category_conf.setdefault(cat, []).append(conf)

            df = pd.DataFrame([
                {
                    "Categoría": label_names[cat],
                    "Cantidad": count,
                    "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}",
                }
                for cat, count in category_count.items()
            ])
            st.dataframe(df, use_container_width=True)
            st.bar_chart(df.set_index("Categoría")["Cantidad"])
        else:
            st.info("No se detectaron objetos con los parámetros actuales.")
            st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")

st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
