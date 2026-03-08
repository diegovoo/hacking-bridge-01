import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import os

def visualize_for_presentation(csv_path='resultados_emociones.csv', max_chars=120, umbral=0.05, show_plot=True):
    """
    Lee el archivo de resultados_emociones.csv y genera un gráfico de barras
    altamente visual y estético, diseñado específicamente para ser incluido 
    en una presentación (PowerPoint, Keynote, etc.).
    """
    if not os.path.exists(csv_path):
        print(f"Error: No se encuentra el archivo {csv_path}")
        return
    
    # Leer el csv
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("El archivo CSV está vacío.")
        return
        
    # Tomar la última fila procesada
    latest_data = df.iloc[-1]
    
    # Obtener y acortar la frase
    frase = str(latest_data['Frase'])
    if len(frase) > max_chars:
        frase_corta = textwrap.shorten(frase, width=max_chars, placeholder="...")
    else:
        frase_corta = frase
        
    # Extraer y asegurar tipo numérico
    emociones = list(latest_data.index[1:])
    valores = pd.to_numeric(pd.Series(latest_data.values[1:]), errors='coerce').fillna(0).tolist()
    
    # Filtrar por umbral
    emociones_filtradas = []
    valores_filtrados = []
    
    for em, val in zip(emociones, valores):
        if val > umbral:
            emociones_filtradas.append(em)
            valores_filtrados.append(val)
            
    if not emociones_filtradas:
        print(f"Ninguna emoción supera el umbral de {umbral}.")
        return

    # Ordenar
    datos_ordenados = sorted(zip(emociones_filtradas, valores_filtrados), key=lambda x: x[1])
    emociones_ord = [x[0].capitalize() for x in datos_ordenados]
    valores_ord = [x[1] for x in datos_ordenados]
    
    # ==========================================
    # CREACIÓN DEL GRÁFICO (MODO PRESENTACIÓN)
    # ==========================================
    
    # Usar un estilo limpio por defecto
    plt.style.use('default')
    
    # Colores de fondo modernos
    bg_color = '#F8F9FA'
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    # Paleta de colores atractiva (de tonos más fríos a cálidos/intensos)
    # Usamos cm.plasma o cm.coolwarm pero suavizado
    colores = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(valores_ord)))
    
    # Barras con mayor grosor y sin bordes
    bars = ax.barh(emociones_ord, valores_ord, height=0.6, color=colores, edgecolor='none')
    
    # Añadir valores exactos de forma grande y destacada al final de la barra
    for bar, val in zip(bars, valores_ord):
        ax.text(val + 0.015, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', 
                va='center', ha='left', 
                fontsize=14, fontweight='bold', color='#2C3E50')
                
    # Minimalismo en los ejes (eliminar bordes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Ajuste de ticks (ítems de los ejes)
    ax.tick_params(axis='y', length=0, pad=10, labelsize=14, labelcolor='#34495E')
    ax.tick_params(axis='x', length=0, labelsize=12, labelcolor='#7F8C8D')
    
    # Marcadores en x y rejilla suave
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.grid(axis='x', color='#E5E8E8', linestyle='-', linewidth=1.5, alpha=0.7, zorder=-1)
    
    # Títulos muy destacados y elegantes
    fig.suptitle('Análisis Emocional del Paciente', 
                 fontsize=24, fontweight='bold', color='#2C3E50', ha='center', y=0.98)
    
    frase_formateada = "\n".join(textwrap.wrap(frase_corta, width=80))
    ax.set_title(f'"{frase_formateada}"', 
                 fontsize=14, fontstyle='italic', color='#7F8C8D', pad=25)
    
    plt.tight_layout()
    
    # Ajustar para dejar espacio arriba para los títulos y a la derecha para los textos de las barras
    plt.subplots_adjust(top=0.82, right=0.92)
    
    # Guardar la imagen automáticamente para usar en presentaciones
    output_img = 'visualizacion_presentacion.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=bg_color)
    print(f"\n✅ Gráfico guardado en alta resolución como: {output_img}")
    print("Ideal para insertar directamente en PowerPoint o Canva.")
    
    if show_plot:
        # Mostrar el gráfico por pantalla
        print("Mostrando gráfico en pantalla. Cierra la ventana emergente para continuar...")
        plt.show()

if __name__ == "__main__":
    visualize_for_presentation()
