import json
import re
import csv
import os
from transformers import pipeline

class PediatricEmotionAnalyzer:
    """
    Analizador de emociones enfocado en Psicología Pediátrica.
    Utiliza el modelo roberta-base-go_emotions para extraer 28 emociones.
    """
    def __init__(self):
        # Cargar el modelo. top_k=None asegura que devuelva las 28 emociones posibles.
        print("Cargando modelo NLP... (esto puede tardar unos segundos la primera vez)")
        self.classifier = pipeline(
            task="text-classification", 
            model="SamLowe/roberta-base-go_emotions", 
            top_k=None
        )
        
        # Definición de muletillas comunes en español
        self.muletillas = [r'\beh\b', r'\besto\b', r'\bmmm\b', r'\bemm\b']
        
    def count_muletillas(self, text):
        text_lower = text.lower()
        count = 0
        for m in self.muletillas:
            count += len(re.findall(m, text_lower))
        return count

    def save_to_csv(self, text, emotions_vector, filename="resultados_emociones.csv"):
        emotions_keys = sorted(emotions_vector.keys())
        headers = ["Frase"] + emotions_keys
        
        file_exists = os.path.isfile(filename)
        
        with open(filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            
            row = [text] + [emotions_vector[k] for k in emotions_keys]
            writer.writerow(row)

    def analyze(self, text):
        """
        Analiza el texto y retorna un JSON estructurado con el vector de 28 emociones
        escaladas de 0 a 10, aplicando lógica pediátrica para carga cognitiva y estado físico.
        """
        from deep_translator import GoogleTranslator
        
        # Traducir el texto al inglés para que el modelo RoBERTa (entrenado en inglés) lo entienda
        try:
            text_en = GoogleTranslator(source='es', target='en').translate(text)
        except Exception as e:
            print(f"Error en la traducción: {e}. Usando texto original (los resultados pueden no ser precisos).")
            text_en = text

        # 1. Predicción base (devuelve lista de diccionarios con 'label' y 'score') usando el texto en inglés
        predictions = self.classifier(text_en)[0]
        
        # 2. Construir el vector y escalar (0 - 1)
        emotions_vector = {}
        for pred in predictions:
            # Asegurarse de no exceder 1.0 (por precaución)
            emotions_vector[pred['label']] = min(1.0, pred['score'])
            
        # 3. Lógica de Muletillas (Indicador de carga cognitiva en niños)
        muletillas_count = self.count_muletillas(text)
        cognitive_load = False
        
        if muletillas_count > 3:
            cognitive_load = True
            # Aplicar multiplicador x1.5 a Nerviosismo y Confusión (con tope en 1.0)
            if 'nervousness' in emotions_vector:
                emotions_vector['nervousness'] = min(1.0, emotions_vector['nervousness'] * 1.5)
            if 'confusion' in emotions_vector:
                emotions_vector['confusion'] = min(1.0, emotions_vector['confusion'] * 1.5)
                
        # 4. Estado Físico Inferido (Fatigue y Pain)
        # Nota: Estas etiquetas no están por defecto en GoEmotions (que tiene 28, todas psicológicas).
        # Por lo tanto, hacemos una inferencia heurística simple buscando palabras clave, o 
        # basándonos en otras emociones si aplicara.
        physical_state = {
            "fatigue": 0.0,
            "pain": 0.0
        }
        
        # Inferencia por keywords (opcional, ajustado al español)
        text_lower = text.lower()
        if re.search(r'\b(duele|dolor|daño|pupita)\b', text_lower):
            physical_state["pain"] = 0.5 # Valor inferido base
        if re.search(r'\b(cansado|sueño|agotado|bostezo)\b', text_lower):
            physical_state["fatigue"] = 0.5 # Valor inferido base
            
        # 5. Estructurar el Salida para Dashboard
        report = {
            "text_analyzed": text,
            "pediatric_indicators": {
                "muletillas_count": muletillas_count,
                "high_cognitive_load": cognitive_load
            },
            "inferred_physical_state": physical_state,
            "emotions_vector": emotions_vector, # Las 28 emociones con su puntuación
            "top_3_emotions": self._get_top_emotions(emotions_vector, 3)
        }
        self.save_to_csv(text, emotions_vector)
        
        return json.dumps(report, indent=4, ensure_ascii=False)
        
    def _get_top_emotions(self, emotions_dict, top_n=3):
        # Ordenamos el diccionario de mayor a menor puntuación
        sorted_emotions = sorted(emotions_dict.items(), key=lambda item: item[1], reverse=True)
        return dict(sorted_emotions[:top_n])


# Ejemplo de uso:
if __name__ == "__main__":
    analyzer = PediatricEmotionAnalyzer()
    
    # Texto de prueba simulando a un niño con carga cognitiva y algo de miedo
    texto_prueba = "¡No! ¡Que no! Es que... jo... siempre es lo mismo. Me duele... me duele el pinchazo de ayer y... y estoy harto de estar aquí encerrado. ¡No es justo! Eh... déjame solo un rato, ¿vale? Por favor..."
    
    print("\n--- Analizando ---")
    print(f"Texto: '{texto_prueba}'\n")
    
    json_result = analyzer.analyze(texto_prueba)
    print(json_result)
