import streamlit as st

st.set_page_config(page_title="Visión por Computador", page_icon="📷", layout="wide")

paginas = [
    st.Page("paginas/teachable.py", title="Teachable Machine", icon="🧠", default=True),
    st.Page("paginas/yolo.py", title="Detección YOLO", icon="🔍"),
]

st.navigation(paginas).run()
