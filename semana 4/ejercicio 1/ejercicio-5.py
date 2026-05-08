import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime, timezone
from ultralytics import YOLO
from supabase import create_client, Client

# --- 1. Configuración de la Página ---
st.set_page_config(page_title="Control Marítimo OBB", layout="wide")
st.title("🚢 Monitoreo de Tráfico Marítimo (OBB)")

# Sidebar para estadísticas
st.sidebar.header("Conteo de Navegación")
north_metric = st.sidebar.empty()
south_metric = st.sidebar.empty()
total_metric = st.sidebar.empty()

# --- 2. Inicialización de Supabase ---
SUPABASE_URL = 'https://ypktexzpugbqvglqirbb.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlwa3RleHpwdWdicXZnbHFpcmJiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzg1NzUsImV4cCI6MjA5Mzc1NDU3NX0.LyHbAY-juJnladWY97H2BV8xSSkd0g-9aPCOToWbLQ8' # Usa la que ya tienes funcionando

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- 3. Carga del Modelo OBB ---
@st.cache_resource
def load_model():
    # El modelo yolo11n-obb o yolo8n-obb es ideal para barcos
    return YOLO('yolo11n-obb.pt') 

model = load_model()

# --- 4. Lógica de Dirección y Subida ---
LINE_Y = 300 # Posición de la línea virtual (ajustar según cámara)

def upload_boat_data(total, north, south):
    try:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_boats": total,
            "north_count": north,
            "south_count": south
        }
        supabase.table("boat_traffic").insert(data).execute()
        print(f"🚀 [SUPABASE] Norte: {north} | Sur: {south} | Total: {total}")
    except Exception as e:
        print(f"❌ Error Supabase: {e}")

# --- 5. Bucle Principal ---
def main():
    frame_placeholder = st.empty()
    run = st.checkbox('Iniciar Radar Marítimo', value=True)
    
    cap = cv2.VideoCapture(0) # Cambiar por URL de IP Cam si es necesario
    
    # Tracking de barcos
    track_history = {} # ID: última posición Y
    conteo_norte = 0
    conteo_sur = 0
    ids_registrados = set()
    
    last_upload_time = 0

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Inferencia OBB con Tracking
        results = model.track(frame, persist=True, conf=0.3, verbose=False)[0]
        
        # Dibujar línea de control (Roja)
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 2)
        cv2.putText(frame, "LINEA DE CONTEO", (10, LINE_Y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if results.obb is not None and results.obb.id is not None:
            # Extraer datos de OBB
            boxes = results.obb.xyxyxyxy.cpu().numpy() # 4 puntos por caja
            ids = results.obb.id.cpu().numpy().astype(int)
            
            for box, obj_id in zip(boxes, ids):
                # Calcular el centro del barco (promedio de los 4 puntos)
                center_y = int(np.mean(box[:, 1]))
                center_x = int(np.mean(box[:, 0]))

                # Lógica de cruce de línea
                if obj_id in track_history:
                    prev_y = track_history[obj_id]
                    
                    if obj_id not in ids_registrados:
                        # Cruzando hacia ARRIBA (Norte)
                        if prev_y > LINE_Y and center_y <= LINE_Y:
                            conteo_norte += 1
                            ids_registrados.add(obj_id)
                            print(f"✨ Barco {obj_id} hacia el NORTE")
                        
                        # Cruzando hacia ABAJO (Sur)
                        elif prev_y < LINE_Y and center_y >= LINE_Y:
                            conteo_sur += 1
                            ids_registrados.add(obj_id)
                            print(f"✨ Barco {obj_id} hacia el SUR")

                track_history[obj_id] = center_y

                # Dibujar OBB (Polígono de 4 puntos)
                pts = box.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID:{obj_id}", (center_x, center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Actualizar Interfaz
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
        
        total_total = conteo_norte + conteo_sur
        north_metric.metric("Hacia el Norte ⬆️", conteo_norte)
        south_metric.metric("Hacia el Sur ⬇️", conteo_sur)
        total_metric.metric("Total Detectados", total_total)

        # Subida periódica a Supabase
        curr_time = time.time()
        if curr_time - last_upload_time > 15: # Cada 15 seg
            if total_total > 0:
                upload_boat_data(total_total, conteo_norte, conteo_sur)
                last_upload_time = curr_time

    cap.release()

if __name__ == "__main__":
    main()