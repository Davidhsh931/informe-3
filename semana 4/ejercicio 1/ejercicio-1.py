import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from supabase import create_client, Client
import time
import asyncio

# --- 1. Configuración de Supabase ---
SUPABASE_URL = "https://ypktexzpugbqvglqirbb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlwa3RleHpwdWdicXZnbHFpcmJiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzg1NzUsImV4cCI6MjA5Mzc1NDU3NX0.LyHbAY-juJnladWY97H2BV8xSSkd0g-9aPCOToWbLQ8"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. Función de Registro (Síncrona para Streamlit) ---
def register_aforo(ambiente: str, person_count: int):
    try:
        supabase.table('aforo_events').insert({
            'ambiente': ambiente,
            'person_count': person_count
        }).execute()
        print(f"✅ Supabase: {person_count} personas en {ambiente}")
    except Exception as e:
        print(f"❌ Error Supabase: {e}")

# --- 3. Interfaz de Streamlit ---
st.set_page_config(page_title="Control de Aforo AgroIA", layout="wide")
st.title("👥 Monitoreo de Aforo en Tiempo Real")

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    AFORO_MAXIMO = st.number_input("Cantidad máxima permitida", min_value=1, value=5)
    ambiente_nombre = st.text_input("Nombre del Ambiente", value="Sala de Manufactura")
    upload_interval = st.slider("Intervalo de registro (seg)", 5, 60, 10)

col_video, col_info = st.columns([2, 1])
frame_placeholder = col_video.empty()
status_placeholder = col_info.empty()

stop_button = st.button("Detener Sistema")

# --- 4. Lógica de Detección ---
model = YOLO('yolov8n.pt') 
cap = cv2.VideoCapture(0)
last_upload_time = 0 

print(f"🚀 Monitor iniciado. Límite: {AFORO_MAXIMO}")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("No se pudo acceder a la cámara.")
        break

    # Detección
    results = model(frame, stream=True, verbose=False)
    person_count = 0

    for r in results:
        if r.boxes:
            classes = r.boxes.cls.cpu().numpy()
            person_count = int((classes == 0).sum())
        annotated_frame = r.plot()

    # --- LÓGICA DE ALERTA VISUAL ---
    color_text = (0, 255, 0) # Verde
    if person_count >= AFORO_MAXIMO:
        color_text = (0, 0, 255) # Rojo
        # Barra de alerta en el frame
        cv2.rectangle(annotated_frame, (0, 0), (frame.shape[1], 60), (0, 0, 255), -1)
        cv2.putText(annotated_frame, "¡ALERTA: AFORO MAXIMO!", (50, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    # Mostrar en Streamlit
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    # Mostrar Info en columna lateral
    with status_placeholder.container():
        if person_count >= AFORO_MAXIMO:
            st.error(f"### 🚨 AFORO EXCEDIDO: {person_count}")
        else:
            st.success(f"### ✅ Aforo Normal: {person_count}")
        
        st.metric("Personas Actuales", person_count, delta=person_count - AFORO_MAXIMO, delta_color="inverse")
        st.write(f"Límite permitido: **{AFORO_MAXIMO}**")

    # --- LÓGICA DE ENVÍO A SUPABASE ---
    current_time = time.time()
    if current_time - last_upload_time > upload_interval:
        # En Streamlit ejecutamos el registro directo o en hilo para no trabar
        register_aforo(ambiente_nombre, person_count)
        last_upload_time = current_time

cap.release()
st.write("Sistema detenido.")