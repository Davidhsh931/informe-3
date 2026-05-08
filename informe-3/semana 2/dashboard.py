import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client
import time

# 1. CONFIGURACIÓN
url = "https://afftlofezngahioexfhy.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFmZnRsb2Zlem5nYWhpb2V4Zmh5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY5NzQ4NTEsImV4cCI6MjA5MjU1MDg1MX0.dtsgQkzqKLBfhfmIHbtQG1f0_chw_yVjnXNmcVaZQ40"
supabase = create_client(url, key)

st.set_page_config(page_title="Mundo-Yacus Monitor", layout="wide", page_icon="🚀")

# --- CSS ADAPTATIVO AVANZADO ---
st.markdown("""
    <style>
    /* Contenedores de métricas con bordes adaptativos */
    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 20px;
        border-radius: 12px;
        background-color: rgba(128, 128, 128, 0.05);
    }
    /* Estilo para las imágenes y sus etiquetas */
    .img-card {
        padding: 12px; 
        border-radius: 10px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .img-card:hover {
        transform: translateY(-5px);
        border-color: #10B981;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4300/4300058.png", width=70)
    st.title("Control Panel")
    auto_refresh = st.checkbox('Auto-refresco (30s)', value=True)
    if st.button('🔄 Refrescar ahora'):
        st.rerun()
    st.divider()
    st.caption("v3.2 | Adaptative UI")

# --- HEADER ---
st.title("🏭 Monitoreo Mundo-Yacus")
st.caption(f"Última actualización: {time.strftime('%H:%M:%S')} | Estado del Sistema: Operativo")

try:
    response = supabase.table("detecciones").select("*").order("created_at", desc=True).execute()
    data = response.data

    if not data:
        st.warning("Esperando conexión con el sensor de cámara...")
    else:
        df = pd.DataFrame(data)
        df['conducta'] = df['conducta'].str.replace(r'^\d+\s*', '', regex=True)
        df['created_at'] = pd.to_datetime(df['created_at'])

        # --- MÉTRICAS SUPERIORES ---
        m1, m2, m3, m4 = st.columns(4)
        
        # Cálculo de fallas recientes
        fallas_hoy = len(df[df['conducta'].str.contains("falla", case=False)])
        
        m1.metric("Eventos Totales", len(df))
        
        estado = df.iloc[0]['conducta']
        m2.metric("Estado Actual", estado, 
                  delta="OK" if "sana" in estado.lower() else "ALERT",
                  delta_color="normal" if "sana" in estado.lower() else "inverse")
        
        # Confianza con comparación al anterior
        conf_actual = df.iloc[0]['confianza']
        conf_previa = df.iloc[1]['confianza'] if len(df) > 1 else conf_actual
        m3.metric("Confianza de IA", f"{conf_actual:.1%}", delta=f"{(conf_actual - conf_previa):.1%}")
        
        m4.metric("Fallas Críticas", fallas_hoy)

        st.divider()

        # --- SECCIÓN DE ANÁLISIS ---
        tab_stats, tab_visual = st.tabs(["📊 Análisis Temporal", "🔍 Galería de Evidencias"])

        with tab_stats:
            col_pie, col_time = st.columns([1, 2])
            
            with col_pie:
                st.write("**Salud de la Maquinaria**")
                fig_pie = px.pie(df, names='conducta', hole=0.6,
                                color='conducta',
                                color_discrete_map={'Maquina sana': '#10B981', 'Maquina falla': '#EF4444'})
                fig_pie.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_time:
                st.write("**Historial de Estabilidad (Confianza vs Tiempo)**")
                # CAMBIO CLAVE: Usamos line con markers para que no se confunda
                fig_time = px.line(df, x='created_at', y='confianza', color='conducta',
                                   markers=True,
                                   color_discrete_map={'Maquina sana': '#10B981', 'Maquina falla': '#EF4444'},
                                   labels={'created_at': 'Tiempo', 'confianza': 'Precisión'})
                
                fig_time.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode="x unified",
                    yaxis=dict(gridcolor='rgba(128, 128, 128, 0.2)', range=[0.5, 1.05])
                )
                st.plotly_chart(fig_time, use_container_width=True)

        with tab_visual:
            st.write("**Historial de Capturas**")
            # Selector moderno en lugar de radio
            filtro = st.segmented_control("Filtrar por categoría:", 
                                          options=["Todas", "Maquina sana", "Maquina falla"], 
                                          default="Todas")
            
            df_v = df if filtro == "Todas" else df[df['conducta'] == filtro]

            for i in range(0, len(df_v), 4):
                cols = st.columns(4)
                for j in range(4):
                    if i + j < len(df_v):
                        row = df_v.iloc[i + j]
                        with cols[j]:
                            st.image(row['imagen_url'], use_container_width=True)
                            
                            is_falla = "falla" in row['conducta'].lower()
                            color_status = "#EF4444" if is_falla else "#10B981"
                            
                            st.markdown(f"""
                                <div class="img-card" style="border-top: 4px solid {color_status};">
                                    <p style="margin:0; font-weight: bold; color: {color_status};">
                                        {'⚠️' if is_falla else '✅'} {row['conducta']}
                                    </p>
                                    <p style="margin:0; font-size: 13px; opacity: 0.8;">
                                        Certeza: {row['confianza']:.2%}
                                    </p>
                                    <code style="font-size: 10px;">{row['created_at'].strftime('%H:%M:%S - %d/%m')}</code>
                                </div>
                                """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error en flujo de datos: {e}")

# Lógica de refresco
if auto_refresh:
    time.sleep(30)
    st.rerun()