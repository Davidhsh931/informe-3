import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import time

# --- 1. CONFIGURACIÓN ---
SUPABASE_URL = "https://ltgjmstozkgvbfvgrkhp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imx0Z2ptc3RvemtndmJmdmdya2hwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc1ODAyNzYsImV4cCI6MjA5MzE1NjI3Nn0.SrEKHPyJe5Zp0cC4lcotunv3CbFXGwATV2tkByOQMW4"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="AgroIA Monitor Pro", layout="wide", page_icon="🥔")

# --- 2. CSS ADAPTATIVO (REPLICA EXACTA) ---
st.markdown("""
    <style>
    /* Fondo oscuro y fuentes */
    .main { background-color: #0e1117; }
    
    /* Contenedores de métricas tipo Mundo-Yacus */
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 10px;
    }
    
    /* Ajuste de texto de métricas */
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 32px !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 16px !important; }
    
    /* Estilo para las pestañas (Tabs) */
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #ffffff; border-bottom-color: #238636; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2153/2153067.png", width=70)
    st.title("Control Panel")
    auto_refresh = st.checkbox('Auto-refresco (30s)', value=True)
    if st.button('🔄 Refrescar ahora'):
        st.rerun()
    st.divider()
    st.caption("v3.5 | Adaptative UI")

# --- 4. HEADER ---
st.title("🏭 Monitoreo AgroIA")
st.caption(f"Última actualización: {time.strftime('%H:%M:%S')} | Estado del Sistema: Operativo")

try:
    # Obtener datos de historial_papas
    response = supabase.table("historial_papas").select("*").order("fecha", desc=True).execute()
    data = response.data

    if not data:
        st.warning("Esperando conexión con el sensor de cámara...")
    else:
        df = pd.DataFrame(data)
        df['fecha'] = pd.to_datetime(df['fecha'])

        # --- MÉTRICAS SUPERIORES (4 Columnas) ---
        m1, m2, m3, m4 = st.columns(4)
        
        # Estado Actual
        estado_act = df.iloc[0]['estado']
        is_sana = "sana" in estado_act.lower()
        
        # Confianza con delta
        conf_act = df.iloc[0]['confianza']
        conf_prev = df.iloc[1]['confianza'] if len(df) > 1 else conf_act
        delta_conf = conf_act - conf_prev

        m1.metric("Eventos Totales", len(df))
        
        m2.metric("Estado Actual", estado_act, 
                  delta="OK" if is_sana else "ALERT",
                  delta_color="normal" if is_sana else "inverse")
        
        m3.metric("Confianza de IA", f"{conf_act:.1f}%", 
                  delta=f"{delta_conf:+.1f}%")
        
        fallas_criticas = len(df[df['estado'] != 'Sana'])
        m4.metric("Fallas Críticas", fallas_criticas)

        st.divider()

        # --- SECCIÓN DE ANÁLISIS ---
        tab_stats, tab_history = st.tabs(["📊 Análisis Temporal", "📋 Historial de Datos"])

        with tab_stats:
            col_pie, col_time = st.columns([1, 2])
            
            with col_pie:
                st.write("**Salud del Cultivo**")
                fig_pie = px.pie(df, names='estado', hole=0.6,
                                color='estado',
                                color_discrete_map={'Sana': '#10B981', 'Tardia': '#EF4444', 'Temprana': '#F59E0B'})
                fig_pie.update_layout(
                    showlegend=False, 
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_time:
                st.write("**Historial de Estabilidad (Confianza vs Tiempo)**")
                fig_time = px.line(df, x='fecha', y='confianza', color='estado',
                                   markers=True,
                                   color_discrete_map={'Sana': '#10B981', 'Tardia': '#EF4444', 'Temprana': '#F59E0B'},
                                   labels={'fecha': 'Tiempo', 'confianza': 'Precisión'})
                
                fig_time.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode="x unified",
                    font=dict(color="#8b949e"),
                    yaxis=dict(gridcolor='rgba(128, 128, 128, 0.1)', range=[0, 105])
                )
                st.plotly_chart(fig_time, use_container_width=True)

        with tab_history:
            st.write("**Registro de Detecciones**")
            # Tabla estilizada
            st.dataframe(
                df[['fecha', 'estado', 'confianza']].head(25),
                use_container_width=True,
                hide_index=True
            )

except Exception as e:
    st.error(f"Error en flujo de datos: {e}")

# Lógica de refresco
if auto_refresh:
    time.sleep(30)
    st.rerun()