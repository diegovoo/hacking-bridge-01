import cv2
import csv
import os
from datetime import datetime
from fer.fer import FER
import warnings

# Suppress standard warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

import sys

def main():
    # --- User Input Prompt ---
    # First, check if user name is provided via command-line arguments
    if len(sys.argv) > 1:
        usuario_id = sys.argv[1].strip()
    else:
        usuario_id = input("Ingrese el nombre del usuario (Enter user name): ").strip()
        
    if not usuario_id:
        usuario_id = "Desconocido"
        
    csv_filename = f"{usuario_id}.csv"

    print("Initializing FER (Facial Expression Recognition)...")
    emotion_detector = FER(mtcnn=True) 
    print("Model loaded successfully!")

    # --- UPDATED CSV SETUP ---
    file_exists = os.path.isfile(csv_filename)
    mode = 'a' if file_exists else 'w'
    
    csv_file = open(csv_filename, mode=mode, newline='')
    csv_writer = csv.writer(csv_file)
    
    # Define the exact order of emotions to guarantee CSV column consistency
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    if not file_exists:
        # Write the new expanded headers
        headers = ["Usuario", "TimeStamp"] + emotion_labels + ["Dominant_Emotion"]
        csv_writer.writerow(headers)
        print(f"Created new log file: {csv_filename}")
    else:
        print(f"Found existing log. Appending data to: {csv_filename}")
    # -------------------------

    # Start video capture (using your external camera index)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    print("Starting webcam feed... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Mirror the image horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)

        # detect_emotions analyzes the frame
        results = emotion_detector.detect_emotions(frame)

        for face in results:
            x, y, w, h = face["box"]
            emotions = face["emotions"] # This is the full dictionary of all 7 scores
            
            # Find the winner for the bounding box text
            top_emotion = max(emotions, key=emotions.get)
            top_score = emotions[top_emotion]

            # --- NEW: LOGGING ALL EMOTIONS ---
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Start building the row with the user and time
            row_data = [usuario_id, current_time]
            
            # Append the 7 individual scores in the exact order we defined in our headers
            for label in emotion_labels:
                row_data.append(emotions.get(label, 0.0))
                
            # Slap the winning emotion at the very end for easy reference
            row_data.append(top_emotion)
            
            # Write the complete 10-column row to the file
            csv_writer.writerow(row_data)
            # ---------------------------------

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{top_emotion.upper()}: {top_score:.2f}"
            text_y = max(10, y - 10) 
            cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the live video feed
        cv2.imshow('FER Real-time Emotion Classifier', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Session saved. Full probability distribution written to {csv_filename}")

if __name__ == "__main__":
    main()