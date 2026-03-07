import cv2
from deepface import DeepFace
import warnings

# Suppress standard warnings
warnings.filterwarnings("ignore")

def main():
    print("Initializing DeepFace Model...")
    
    # Start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Starting webcam feed... Press 'q' to quit.")

    # --- PERFORMANCE VARIABLES ---
    frame_count = 0
    process_every_n_frames = 3 # Increase this if your CPU struggles, decrease for faster updates
    last_faces = [] # Stores the results to draw boxes between processed frames

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame_count += 1

        # Only run the heavy deep learning model every Nth frame
        if frame_count % process_every_n_frames == 0:
            try:
                # Analyze the frame. 
                # detector_backend="opencv" is the fastest. If you want better accuracy 
                # (and no 'tight crop' issues), change it to "mtcnn" or "retinaface".
                results = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=True, # Forces DeepFace to confirm a face exists
                    detector_backend="mtcnn",
                    silent=True # Prevents DeepFace from spamming your terminal with progress bars
                )
                
                # DeepFace returns a dict for one face, or a list of dicts for multiple faces.
                # We normalize it to always be a list for easier iteration below.
                if not isinstance(results, list):
                    results = [results]
                    
                last_faces = results
                
            except ValueError:
                # DeepFace throws a ValueError if no face is detected in the frame.
                # Catch it and clear the last known faces so bounding boxes disappear.
                last_faces = []
            except Exception as e:
                print(f"Unexpected error during analysis: {e}")

        # Draw the bounding boxes and text based on the last successful analysis
        for face in last_faces:
            region = face['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            # Extract the emotion data
            dominant_emotion = face['dominant_emotion']
            
            # DeepFace returns emotion scores as percentages (0-100)
            score = face['emotion'][dominant_emotion]

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Prepare and display the prediction text
            text = f"{dominant_emotion.upper()}: {score:.1f}%"
            text_y = max(10, y - 10) 
            cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Show the live video feed
        cv2.imshow('DeepFace Real-time Emotion Classifier', frame)

        # Listen for the 'q' key to stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Release resources cleanly
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()