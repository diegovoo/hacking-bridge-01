import streamlit as st
import json
import os
from PIL import Image
from emotion_analyzer import PediatricEmotionAnalyzer
from visualizador_presentacion import visualize_for_presentation

# Configurar la página de Streamlit
st.set_page_config(
    page_title="Analizador y Visualizador de Emociones",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Interfaz de Análisis Emocional Pediátrico")
st.markdown("Esta interfaz permite analizar textos para detectar emociones pediátricas y luego generar gráficos de presentación basados en los resultados.")

# Cachar el modelo para evitar recargarlo cada vez que se interactúa con la interfaz
@st.cache_resource
def load_analyzer():
    return PediatricEmotionAnalyzer()

try:
    analyzer = load_analyzer()
except Exception as e:
    st.error(f"Error al cargar el analizador: {e}")
    st.stop()

# Crear pestañas para separar las dos funcionalidades
tab_analyzer, tab_visualizer = st.tabs(["1️⃣ Analizador de Emociones", "2️⃣ Visualizador para Presentaciones"])

with tab_analyzer:
    st.header("Análisis de Texto")
    
    texto_default = "¡No! ¡Que no! Es que... jo... siempre es lo mismo. Me duele... me duele el pinchazo de ayer y... y estoy harto de estar aquí encerrado. ¡No es justo! Eh... déjame solo un rato, ¿vale? Por favor..."
    texto_input = st.text_area("Introduce el texto del paciente pediátrico a analizar:", value=texto_default, height=150)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analizar_btn = st.button("Analizar Texto", type="primary")
        
    if analizar_btn:
        if texto_input.strip() == "":
            st.warning("Por favor, introduce un texto para analizar.")
        else:
            with st.spinner("Analizando texto y extrayendo emociones... (esto puede tardar unos segundos)"):
                try:
                    # El analizador guarda automáticamente en resultados_emociones.csv
                    json_result = analyzer.analyze(texto_input)
                    resultado_dict = json.loads(json_result)
                    
                    st.success("¡Análisis completado y guardado con éxito!")
                    
                    # Mostrar resultados de forma estructurada
                    st.subheader("Resultados Principales")
                    
                    top_emotions = resultado_dict.get("top_3_emotions", {})
                    cols = st.columns(len(top_emotions))
                    for i, (emocion, score) in enumerate(top_emotions.items()):
                        with cols[i]:
                            st.metric(label=emocion.capitalize(), value=f"{score*100:.1f}%")
                            
                    st.subheader("Reporte Completo (JSON)")
                    st.json(resultado_dict)
                    
                except Exception as e:
                    st.error(f"Error durante el análisis: {e}")

with tab_visualizer:
    st.header("Generador de Gráficos de Presentación")
    st.markdown("Genera una visualización estética basada en el **último análisis realizado** (guardado en `resultados_emociones.csv`).")
    
    csv_path = 'resultados_emociones.csv'
    
    # Controles para la visualización
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        umbral_val = st.slider(
            "Umbral mínimo de emoción (oculta emociones con puntuación menor):", 
            min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
    with col_v2:
        max_chars_val = st.number_input(
            "Máximo de caracteres de la frase a mostrar en el título:",
            min_value=50, max_value=300, value=120, step=10
        )
        
    generar_btn = st.button("Generar Gráfico", type="primary")
    
    # Contenedor para mostrar la imagen generada
    image_container = st.empty()
    
    # Mostrar la imagen actual si el botón fue presionado
    if generar_btn:
        if not os.path.exists(csv_path):
            st.error(f"No se encuentra el archivo `{csv_path}`. Por favor, analiza un texto primero en la otra pestaña.")
        else:
            with st.spinner("Generando gráfico de alta resolución..."):
                try:
                    # Llamar a la función con el parche de show_plot=False
                    visualize_for_presentation(csv_path=csv_path, max_chars=max_chars_val, umbral=umbral_val, show_plot=False)
                    
                    img_path = 'visualizacion_presentacion.png'
                    if os.path.exists(img_path):
                        image_container.image(Image.open(img_path), caption="Gráfico generado listo para usar en presentaciones.")
                        
                        # Botón para descargar la imagen
                        with open(img_path, "rb") as file:
                            st.download_button(
                                label="Descargar Gráfico",
                                data=file,
                                file_name="visualizacion_emociones.png",
                                mime="image/png"
                            )
                    else:
                        st.error("No se pudo generar la imagen. Verifica que haya emociones que superen el umbral.")
                except Exception as e:
                    st.error(f"Error al generar el gráfico: {e}")
