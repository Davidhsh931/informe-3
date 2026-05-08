import streamlit as st
import cv2
from ultralytics import YOLO
from supabase import create_client, Client
from collections import Counter
import matplotlib.pyplot as plt
import time

# --- 1. Configuración de Supabase ---
url = "https://ypktexzpugbqvglqirbb.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlwa3RleHpwdWdicXZnbHFpcmJiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzg1NzUsImV4cCI6MjA5Mzc1NDU3NX0.LyHbAY-juJnladWY97H2BV8xSSkd0g-9aPCOToWbLQ8" 
supabase: Client = create_client(url, key)

# --- 2. Configuración de la Interfaz Web ---
st.set_page_config(page_title="Monitor Agro IA Pro", layout="wide")
st.title("🐄 Clasificación de Animales en Tiempo Real")

col_video, col_stats = st.columns([2, 1])
frame_placeholder = col_video.empty()
chart_placeholder = col_stats.empty()
stop_button = st.button("Detener Sistema")

# --- 3. Lógica de Detección e IA ---
model = YOLO('yolov8s.pt') 
target_class_ids = [15, 16, 17, 18, 19]

cap = cv2.VideoCapture(0)

# Ajuste para que el primer envío sea inmediato
last_upload_time = 0 
upload_interval = 10 

print("🚀 Sistema Iniciado. Esperando detecciones...")

while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, stream=True, verbose=False, conf=0.45, classes=target_class_ids, imgsz=640)
    
    detected_names = []
    annotated_frame = frame.copy()

    for r in results:
        annotated_frame = r.plot()
        if r.boxes:
            for c in r.boxes.cls:
                name = model.names[int(c)]
                detected_names.append(name)

    counts = Counter(detected_names)

    # IMPRESIÓN EN TERMINAL (Cada frame que detecta algo)
    if counts:
        print(f"🔍 Detectado en cámara: {dict(counts)}", end="\r")

    # Visualización en Streamlit
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    if counts:
        with chart_placeholder.container():
            st.write(f"### Detecciones: {sum(counts.values())}")
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
            ax.axis('equal')
            st.pyplot(fig)
            plt.close(fig)
            for animal, q in counts.items():
                st.info(f"📍 {animal.capitalize()}: {q}")
    else:
        chart_placeholder.warning("Buscando animales...")

    # --- 5. Sincronización con Supabase ---
    current_time = time.time()
    if counts and (current_time - last_upload_time > upload_interval):
        print(f"\n      ") # Limpiar línea de terminal
        print(f"📤 Enviando a Supabase...")
        for animal, q in counts.items():
            try:
                supabase.table('animal_counts').insert({
                    "animal_type": animal,
                    "count": q,
                    "location": "Sede Central - AgroIA"
                }).execute()
                print(f"✅ DB Update: {animal.upper()} | Cantidad: {q}")
            except Exception as e:
                print(f"❌ Error en DB: {e}")
        last_upload_time = current_time

    time.sleep(0.01)

cap.release()
print("\n🛑 Sistema detenido por el usuario.")
st.success("Sistema finalizado.")