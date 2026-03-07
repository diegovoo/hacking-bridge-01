import cv2
import csv
import os
from datetime import datetime
from fer.fer import FER
import warnings

# Suppress standard warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

def main():
    # --- NEW: User Input Prompt ---
    # We ask for the name before loading the heavy AI model so the user doesn't have to wait
    usuario_id = input("Ingrese el nombre del usuario (Enter user name): ").strip()
    
    # Fallback just in case the user hits Enter without typing a name
    if not usuario_id:
        usuario_id = "Desconocido"
        
    csv_filename = f"{usuario_id}.csv"
    # ------------------------------

    print("Initializing FER (Facial Expression Recognition)...")
    emotion_detector = FER(mtcnn=True) 
    print("Model loaded successfully!")

    # --- CSV SETUP WITH APPEND LOGIC ---
    # Check if the file already exists in the directory
    file_exists = os.path.isfile(csv_filename)
    
    # Use 'a' (append) if it exists, otherwise 'w' (write)
    mode = 'a' if file_exists else 'w'
    
    csv_file = open(csv_filename, mode=mode, newline='')
    csv_writer = csv.writer(csv_file)
    
    if not file_exists:
        # File is new, so we write the headers
        csv_writer.writerow(["Usuario", "TimeStamp", "Emocion", "Confianza_en_prediccion"])
        print(f"Created new log file: {csv_filename}")
    else:
        # File exists, skip headers and just append
        print(f"Found existing log. Appending data to: {csv_filename}")
    # -----------------------------------

    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam. Please ensure it is connected.")
        return

    print("Starting webcam feed... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # detect_emotions analyzes the frame
        results = emotion_detector.detect_emotions(frame)

        for face in results:
            # 1. Extract bounding box coordinates
            x, y, w, h = face["box"]
            emotions = face["emotions"]
            
            # 2. Find the emotion with the highest confidence score
            top_emotion = max(emotions, key=emotions.get)
            top_score = emotions[top_emotion]

            # --- CSV LOGGING ---
            # Grab the current time down to the millisecond
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # Write the dynamic User ID, exact time, emotion, and score to the file
            csv_writer.writerow([usuario_id, current_time, top_emotion, top_score])
            # -------------------

            # 3. Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # 4. Prepare and display the prediction text
            text = f"{top_emotion.upper()}: {top_score:.2f}"
            
            # Ensure the text doesn't draw off the top edge of the screen
            text_y = max(10, y - 10) 
            cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the live video feed
        cv2.imshow('FER Real-time Emotion Classifier', frame)

        # Listen for the 'q' key to stop the loop cleanly
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Release resources cleanly
    cap.release()
    cv2.destroyAllWindows()
    
    # Close the CSV file safely
    csv_file.close()
    print(f"Session saved. Data securely written to {csv_filename}")

if __name__ == "__main__":
    main()