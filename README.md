<div align="center">
  <h1>🎮 Hacking Bridge: Plataforma de Evaluación Emocional Infantil 🧠</h1>
  <p>Evaluador de emociones infantiles antes y después del juego utilizando Inteligencia Artificial.</p>
  
  [![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
  [![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
  [![FER](https://img.shields.io/badge/Computer_Vision-FER-green.svg)]()
</div>

## 🎥 Demo del Analizador Textual

![Demo Analizando Texto y Emociones](demo-analisis.webp)

## 📖 Sobre el Proyecto

Este repositorio fue desarrollado como solución algorítmica y visual para la **hackathon de Hacking Bridge**. 
El reto propuesto era claro: **¿Cómo documentar y analizar de forma objetiva con IA los sentimientos y emociones de los niños antes y después de participar en dinámicas de juego?**

Utilizando modelos de procesamiento de lenguaje natural y visión computacional, el prototipo es capaz de analizar indicadores de expresión facial y verbal, evaluando un cambio en el estado anímico, fatiga y estrés cognitivo.

---

## 🚀 Modelos de Inteligencia Artificial Utilizados

El sistema extrae conclusiones combinando los datos de dos fuentes distintas:

### 1. 📝 Análisis Textual (Transcripciones de Audio)
- **Modelo NLP:** `SamLowe/roberta-base-go_emotions` (Implementado vía Hugging Face `transformers`).
- **Funcionamiento:** Capaz de recibir texto o transcripciones derivadas de las verbalizaciones del niño y catalogarlas dentro de un espectro complejo de **28 emociones distintas**.
- **Módulo de Lógica Pediátrica:**
  - **Carga Cognitiva:** Detecta automáticamente el conteo de *muletillas verbales* (como "eh", "mmm", "esto"). Si estas exceden un límite, el sistema interpreta alta "carga cognitiva", multiplicando de forma algorítmica el peso del *nerviosismo* y la *confusión* detectadas.
  - **Estado Físico:** Ejecuta inferencias heurísticas para detectar indicadores de **dolor** o **fatiga** física basándose en las frases expresadas por el niño.
- **Flujo de Idioma:** Dado que el modelo es nativo en inglés, utiliza `deep-translator` en segundo plano para procesar respuestas de los niños hispanohablantes a tiempo real sin perder significados de peso.

### 2. 📹 Análisis Facial Biológico en Tiempo Real
- **Modelo de Visión:** `FER` (Facial Expression Recognition) apoyado en la arquitectura **MTCNN** (Multi-task Cascaded Convolutional Networks) para la precisa detección de rostros.
- **Funcionamiento:** Extrae fotogramas en vivo a través de la webcam para predecir al instante un desglose probabilístico de las **7 emociones humanas básicas** (Felicidad, Tristeza, Enojo, Sorpresa, Disgusto, Miedo, Neutralidad).
- **Dashboard Automático:** Genera un registro en `.csv` segundo a segundo permitiendo una visualización de progreso con una librería generadora de plots estáticos que ayudan a los evaluadores a entender fácilmente la "línea del tiempo" de la sesión.

---

## 🛠️ Detalles Técnicos y Estructura

El ecosistema se orquesta bajo una interfaz central utilizando **Streamlit** y Python puro.

- `app.py`: Script central. Se encarga de mostrar la UI de pestañas, y la invocación de subprocesos aislados.
- `analisis_facial/fer_todos_datos.py`: Corre en un *subprocess* aislado para no bloquear el puerto principal. Se encarga de abrir la señal de video y grabar los tensores devueltos por la red convolucional.
- `analisis_texto/emotion_analyzer.py`: Clase de Python orientada a objetos embebida en memoria que procesa las cadenas de caracteres y devuelve diccionarios e insights estructurados.

---

## ⚙️ Cómo Ejecutar el Proyecto

### Pasos Iniciales
Clona el repositorio en tu ordenador local:
```bash
git clone <url-del-repositorio>
cd hacking-bridge-01
```

### Configuración del Entorno Virtual
Para no causar conflictos instalando librerías científicas, se recomienda fervientemente utilizar el entorno conda listo en `environment.yml` que ya contiene versiones compatibles de las dependencias (`tensorflow`, `tf-keras`, `transformers`, `torch`, `fer`, etc).

```bash
conda env create -f environment.yml
conda activate hacking-bridge
```

### Lanzamiento
Una vez que el entorno haya sido creado e inicializado con éxito, arranca el panel de mando ejecutando:

```bash
python -m streamlit run app.py
```
> La interfaz del Centro de Evaluación se abrirá en tu navegador de preferencia en `http://localhost:8501`.

---

## 💡 Sugerencia de Metodología de Uso

Para testear si el juego genera un cambio en el sentimiento, un evaluador de comportamiento debería seguir estos simples pasos en la app:

1. **Pre-test:** Llama al niño, ponle nombre o session_ID (ej: `Diego_PRE`). Pídele que se siente delante de la cámara y cuente un poco cómo se siente antes de ir a jugar. Dale al botón de **Analizar Texto** con lo que el niño haya verbalizado y luego al inicio del análisis visual.
2. **🕹️ Hora de Jugar:** ¡Manda a los niños a hacer un juego divertido o de aprendizaje! 
3. **Post-test:** Trae al mismo niño luego de terminar el juego. Graba una nueva sesión llamada (ej: `Diego_POST`). Pídele que hable sobre su experiencia. 
4. **Insights:** Utiliza la herramienta "Generar Gráfico" tanto en los registros visuales como textuales o revisa los archivos `.csv` cruzando datos de antes y después.

---

<div align="center">
  <i>Código y algoritmos orientados a poder comprender mejor la mente de los más pequeños. ❤️🧩</i>
</div>
