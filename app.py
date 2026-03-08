import streamlit as st
import subprocess
import os
import sys

# Configure Streamlit page
st.set_page_config(
    page_title="Herramientas de Análisis Psicológico",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Title and Description
st.title("🏥 Centro de Evaluación Emocional")
st.markdown("""
Bienvenido al panel centralizado de herramientas de evaluación psicológica del proyecto **Hacking Bridge**.
Selecciona la herramienta que deseas utilizar desde el menú lateral o las pestañas a continuación.
""")

# Create Tabs for different functionalities
tab1, tab2 = st.tabs(["📹 Análisis Facial en Tiempo Real", "📝 Analizador de Textos y Emociones"])

# ==========================================
# TAB 1: FACIAL ANALYSIS
# ==========================================
with tab1:
    st.header("Análisis de Expresiones Faciales")
    st.markdown("""
    Esta herramienta utiliza la cámara web para detectar y analizar las expresiones faciales en tiempo real.
    Los datos de las 7 emociones básicas se registrarán continuamente en un archivo CSV dentro de la carpeta `analisis_facial`.
    """)
    
    st.info("💡 **Asegúrate de tener la cámara conectada y libre (no en uso por otra app).**")
    
    # Input for Patient/User ID
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        patient_id = st.text_input("Identificador del Paciente/Sesión:", placeholder="Ej: Sujeto_001")
        
    start_facial_btn = st.button("▶️ Iniciar Cámara y Análisis", type="primary")
    
    if start_facial_btn:
        if not patient_id:
            st.error("Por favor, introduce un Identificador de Paciente válido antes de comenzar.")
        else:
            facial_script_path = os.path.join(os.getcwd(), 'analisis_facial', 'fer_todos_datos.py')
            
            if not os.path.exists(facial_script_path):
                st.error(f"No se pudo encontrar el script de análisis facial en: `{facial_script_path}`")
            else:
                st.success(f"Iniciando análisis para el paciente: **{patient_id}**. Se abrirá una nueva ventana con la cámara.")
                st.warning("⚠️ **Para detener la grabación, selecciona la ventana de la cámara y presiona la tecla 'q'.**")
                
                # Launch the script using subprocess in its own directory
                try:
                    # Using sys.executable ensures we use the exact same Python interpreter (conda env)
                    # We pass the patient ID directly to our newly modified script
                    working_dir = os.path.join(os.getcwd(), 'analisis_facial')
                    subprocess.Popen([sys.executable, 'fer_todos_datos.py', patient_id], cwd=working_dir)
                    
                    st.toast("Proceso facial iniciado en segundo plano", icon="✅")
                except Exception as e:
                    st.error(f"Error al intentar ejecutar el proceso: {e}")
                    
    st.markdown("---")
    st.subheader("📊 Generar Dashboard de Resultados")
    st.markdown("Una vez finalizada la sesión de cámara, puedes generar un informe visual de las emociones detectadas.")
    
    col_g1, col_g2 = st.columns([2, 1])
    with col_g1:
        plot_patient_id = st.text_input("Identificador del Paciente para el Informe:", value=patient_id if 'patient_id' in locals() else "", key="plot_id")
        
    generate_plot_btn = st.button("🖼️ Generar Gráfico de Resultados", type="secondary")
    
    if generate_plot_btn:
        if not plot_patient_id:
            st.warning("Introduce el identificador del paciente para buscar su archivo de datos.")
        else:
            working_dir = os.path.join(os.getcwd(), 'analisis_facial')
            csv_path = os.path.join(working_dir, f"{plot_patient_id}.csv")
            img_output_path = os.path.join(working_dir, f"{plot_patient_id}_dashboard.png")
            
            if not os.path.exists(csv_path):
                st.error(f"No se encontraron datos para el paciente '{plot_patient_id}'. Asegúrate de haber realizado la grabación primero.")
            else:
                with st.spinner(f"Generando dashboard para {plot_patient_id}..."):
                    try:
                        # Call final_plot.py via subprocess to generate the image
                        result = subprocess.run(
                            [sys.executable, 'final_plot.py', f"{plot_patient_id}.csv", "--save-path", f"{plot_patient_id}_dashboard.png"],
                            cwd=working_dir,
                            capture_output=True,
                            text=True
                        )
                        
                        if result.returncode == 0 and os.path.exists(img_output_path):
                            from PIL import Image
                            st.success("Dashboard generado correctamente.")
                            st.image(Image.open(img_output_path), caption=f"Dashboard Emocional - {plot_patient_id}")
                            
                            # Add download button
                            with open(img_output_path, "rb") as file:
                                st.download_button(
                                    label="Descargar Dashboard",
                                    data=file,
                                    file_name=f"{plot_patient_id}_dashboard.png",
                                    mime="image/png",
                                    key="download_facial_plot"
                                )
                        else:
                            st.error(f"Error al generar el gráfico. Puede que el CSV no tenga el formato correcto.")
                            if result.stderr:
                                st.code(result.stderr)
                    except Exception as e:
                        st.error(f"Error ejecutando final_plot.py: {e}")

# ==========================================
# TAB 2: TEXT ANALYSIS (EMBEDDED)
# ==========================================
with tab2:
    # Instead of launching another Streamlit subprocess which creates messy ports,
    # we just import the logic directly from the text analyzer scripts using sys.path
    
    text_analyzer_path = os.path.join(os.getcwd(), 'analisis_texto')
    if text_analyzer_path not in sys.path:
        sys.path.append(text_analyzer_path)
        
    try:
        from analisis_texto.emotion_analyzer import PediatricEmotionAnalyzer
        from analisis_texto.visualizador_presentacion import visualize_for_presentation
        import json
        from PIL import Image
        
        st.header("Análisis de Texto y Extracción de Emociones")
        
        # Cache the text analyzer model
        @st.cache_resource
        def load_text_analyzer():
            return PediatricEmotionAnalyzer()
            
        analyzer = load_text_analyzer()
        
        st.markdown("Analiza transcripciones o textos para extraer 28 emociones y generar gráficos de presentación.")
        
        # Sub-tabs inside Tab 2 for organization
        sub_tab_txt1, sub_tab_txt2 = st.tabs(["Analizar Texto", "Generar Gráfico"])
        
        with sub_tab_txt1:
            texto_default = "¡No! ¡Que no! Es que... jo... siempre es lo mismo. Me duele... me duele el pinchazo de ayer y... y estoy harto de estar aquí encerrado. ¡No es justo! Eh... déjame solo un rato, ¿vale? Por favor..."
            texto_input = st.text_area("Introduce el texto a analizar:", value=texto_default, height=150, key="txt_input")
            
            if st.button("Analizar Texto", type="primary", key="btn_analizar"):
                if texto_input.strip() == "":
                    st.warning("El texto no puede estar vacío.")
                else:
                    with st.spinner("Procesando con modelo NLP..."):
                        # Ensure we save the CSV in the analisis_texto folder to keep it clean
                        csv_output_path = os.path.join(text_analyzer_path, "resultados_emociones.csv")
                        
                        # Temporal monkey-patch of save_to_csv to force correct directory
                        original_save = analyzer.save_to_csv
                        analyzer.save_to_csv = lambda txt, vec, fname=csv_output_path: original_save(txt, vec, fname)
                        
                        json_result = analyzer.analyze(texto_input)
                        resultado_dict = json.loads(json_result)
                        
                        # Restore original method
                        analyzer.save_to_csv = original_save
                        
                        st.success("¡Análisis guardado con éxito!")
                        
                        st.subheader("Emociones Principales")
                        top_emotions = resultado_dict.get("top_3_emotions", {})
                        cols = st.columns(max(1, len(top_emotions)))
                        for i, (emocion, score) in enumerate(top_emotions.items()):
                            with cols[i]:
                                st.metric(label=emocion.capitalize(), value=f"{score*100:.1f}%")

        with sub_tab_txt2:
            st.markdown("Genera una visualización estética basada en el último análisis.")
            csv_path = os.path.join(text_analyzer_path, 'resultados_emociones.csv')
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                umbral_val = st.slider("Umbral mínimo de emoción:", 0.0, 1.0, 0.05, 0.01)
            with col_v2:
                max_chars_val = st.number_input("Máx caracteres título:", 50, 300, 120, 10)
                
            if st.button("Generar Gráfico", type="primary", key="btn_grafico"):
                if not os.path.exists(csv_path):
                    st.error("No hay datos analizados. Analiza un texto primero.")
                else:
                    with st.spinner("Generando..."):
                        # Go to the folder temporarily to generate image there
                        img_path = os.path.join(text_analyzer_path, 'visualizacion_presentacion.png')
                        
                        # Call the imported function
                        # Temporarily change directory so it saves the file in the right place as coded there
                        cwd_original = os.getcwd()
                        os.chdir(text_analyzer_path)
                        try:
                            # Note: pass show_plot=False 
                            visualize_for_presentation(csv_path='resultados_emociones.csv', max_chars=max_chars_val, umbral=umbral_val, show_plot=False)
                            os.chdir(cwd_original)
                            
                            if os.path.exists(img_path):
                                st.image(Image.open(img_path))
                        except Exception as e:
                            os.chdir(cwd_original)
                            st.error(f"Error generando gráfico: {e}")
                            
    except ImportError as e:
        st.error(f"No se pudieron cargar los módulos de análisis de texto. Error: {e}")
        st.info("Asegúrate de que la carpeta 'analisis_texto' contenga `emotion_analyzer.py` y `visualizador_presentacion.py`.")

# Sidebar Info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg", width=50) # Just a placeholder decoration
    st.markdown("---")
    st.markdown("### ℹ️ Sobre el Sistema")
    st.markdown("Este panel permite controlar todas las herramientas locales para análisis de expresiones faciales y texto en entornos de psicología pediátrica.")
