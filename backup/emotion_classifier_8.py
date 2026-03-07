import cv2
import torch
import warnings
from transformers import pipeline
from PIL import Image

# Ignore standard warnings for a cleaner output
warnings.filterwarnings("ignore")

def main():
    print("Loading emotion classification model (8 emotions: anger, contempt, disgust, fear, happy, neutral, sad, surprise)...")
    
    # Initialize the Hugging Face pipeline for image classification
    # dima806/facial_emotions_image_detection is trained on AffectNet and supports exactly these 8 emotions.
    classifier = pipeline("image-classification", model="dima806/facial_emotions_image_detection", device=-1) # device=-1 forces CPU
    print("Model loaded successfully!")

    # Initialize OpenCV Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam. Please ensure it is connected and not being used by another application.")
        return

    print("Starting webcam feed... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # Convert the frame to grayscale for OpenCV face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop the face area for emotion classification
            face_crop = frame[y:y+h, x:x+w]
            
            try:
                # Convert BGR (OpenCV format) to RGB (PIL format)
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)

                # Classify emotion on the cropped face
                results = classifier(pil_img)
                
                # Get the top prediction
                top_result = results[0]
                label = top_result['label']
                score = top_result['score']

                # Print the detected emotion in real time
                print(f"Detected Emotion: {label.upper()} ({score:.2f})")

                # Display the prediction on the video feed
                text = f"{label}: {score:.2f}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Error processing face crop: {e}")

        # Show the video feed
        cv2.imshow('Real-time Emotion Classifier (8 Emotions)', frame)

        # Listen for the 'q' key to stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Release resources cleanly
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
