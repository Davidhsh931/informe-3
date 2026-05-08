import cv2
import numpy as np
import json
import time
import streamlit as st
from datetime import datetime
from ultralytics import YOLO
from supabase import create_client, Client

# --- 1. Configuración de Supabase ---
URL = 'https://ypktexzpugbqvglqirbb.supabase.co'
KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlwa3RleHpwdWdicXZnbHFpcmJiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzgxNzg1NzUsImV4cCI6MjA5Mzc1NDU3NX0.LyHbAY-juJnladWY97H2BV8xSSkd0g-9aPCOToWbLQ8'
supabase: Client = create_client(URL, KEY)

# --- 2. Función de envío DIRECTO ---
def send_alert_sync(bbox, zone, label="Persona"):
    try:
        data = {
            'timestamp': datetime.now().isoformat(),
            'image_path': "Live_Camera_Web",
            'zone_coordinates': json.dumps(zone),
            'person_bounding_box': json.dumps(bbox),
            'alert_message': f"⚠️ INTRUSIÓN: {label} detectada"
        }
        supabase.table('alerts').insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Error Supabase: {e}")
        return False

# --- 3. Interfaz Streamlit (Actualizada 2026) ---
st.set_page_config(page_title="IA Security Monitor", layout="wide")
st.title("🛡️ Sistema de Vigilancia con IA (Segmentación)")

col_video, col_alerts = st.columns([2, 1])

with col_alerts:
    st.header("🚨 Historial de Alertas")
    alert_placeholder = st.empty()

with col_video:
    frame_placeholder = st.empty()
    stop_button = st.button("Detener Monitoreo")

# --- 4. Bucle Principal ---
def main():
    # Cargamos modelo de segmentación
    model = YOLO('yolov8n-seg.pt')
    cap = cv2.VideoCapture(0)
    
    # Zona Prohibida
    FORBIDDEN_ZONE = [0, 0, 300, 250] 
    
    last_alert_time = 0
    alert_cooldown = 5 

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret: break

        h, w, _ = frame.shape
        results = model(frame, stream=True, verbose=False)
        
        intrusion_detected = False
        current_bbox = None

        for r in results:
            if r.masks is not None:
                for i, cls_idx in enumerate(r.boxes.cls):
                    if int(cls_idx) == 0:  # Persona
                        m_data = r.masks.data[i].cpu().numpy()
                        person_mask = cv2.resize(m_data, (w, h), interpolation=cv2.INTER_NEAREST)

                        zone_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.rectangle(zone_mask, (FORBIDDEN_ZONE[0], FORBIDDEN_ZONE[1]), 
                                     (FORBIDDEN_ZONE[2], FORBIDDEN_ZONE[3]), 1, -1)

                        if np.any(cv2.bitwise_and(person_mask.astype(np.uint8), zone_mask)):
                            intrusion_detected = True
                            box_data = r.boxes.xyxy[i].cpu().numpy().flatten()
                            current_bbox = [round(float(val), 2) for val in box_data]
            
            frame = r.plot()

        # Dibujar UI de Zona
        color_zona = (0, 0, 255) if intrusion_detected else (0, 255, 0)
        cv2.rectangle(frame, (FORBIDDEN_ZONE[0], FORBIDDEN_ZONE[1]), 
                     (FORBIDDEN_ZONE[2], FORBIDDEN_ZONE[3]), color_zona, 3)
        
        if intrusion_detected:
            cv2.putText(frame, "!!! INTRUSO DETECTADO !!!", (10, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            
            curr_time = time.time()
            if curr_time - last_alert_time > alert_cooldown:
                if send_alert_sync(current_bbox, FORBIDDEN_ZONE):
                    last_alert_time = curr_time
                    # Consultar últimas alertas para mostrar en la web
                    try:
                        res = supabase.table('alerts').select("*").order('timestamp', desc=True).limit(8).execute()
                        with alert_placeholder.container():
                            for alert in res.data:
                                # Diseño de alerta más limpio
                                time_str = alert['timestamp'].split('T')[1][:8]
                                st.toast(f"Nueva intrusión detectada a las {time_str}")
                                st.error(f"**{time_str}** - {alert['alert_message']}")
                    except: pass

        # --- MEJORA: Cambio de use_container_width por width='stretch' ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()
    st.info("Monitor de seguridad detenido.")

if __name__ == "__main__":
    main()