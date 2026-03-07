import cv2
import csv
from datetime import datetime
from fer.fer import FER
import warnings

# Suppress standard warnings for a cleaner terminal output
warnings.filterwarnings("ignore")

def main():
    print("Initializing FER (Facial Expression Recognition)...")
    
    # Initialize the detector. 
    emotion_detector = FER(mtcnn=True) 
    print("Model loaded successfully!")

    # --- CSV SETUP ---
    # Create a unique filename based on the current date and time
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"emotion_log_{timestamp_str}.csv"
    
    # Open the file in write mode and create a CSV writer object
    csv_file = open(csv_filename, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write the header row
    csv_writer.writerow(["Timestamp", "Dominant_Emotion", "Confidence_Score"])
    print(f"Logging data to: {csv_filename}")
    # -----------------

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

            # --- OPTIONAL: Thresholding Fix ---
            # if top_emotion == 'fear' and top_score < 0.70:
            #     top_emotion = 'neutral'
            #     top_score = emotions['neutral']
            # ----------------------------------

            # --- CSV LOGGING ---
            # Grab the current time down to the millisecond
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # Write the exact time, the emotion, and the score to the file
            csv_writer.writerow([current_time, top_emotion, top_score])
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
    print(f"Session saved. Data written to {csv_filename}")

if __name__ == "__main__":
    main()