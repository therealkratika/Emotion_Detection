import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_face(face_gray):
    size = 64 
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    face_gray = clahe.apply(face_gray)
    
    face_resized = cv2.resize(face_gray, (size, size), interpolation=cv2.INTER_CUBIC)
    
    face_normalized = face_resized.astype("float32") / 255.0
    
    return np.expand_dims(np.expand_dims(face_normalized, axis=-1), axis=0)

def main():
    try:
        model = load_model("emotion_model.h5", compile=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    results_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            processed = preprocess_face(face_img)
            
            # Get prediction
            preds = model.predict(processed, verbose=0)[0]

            preds[6] *= 0.5 
            
            # Add to buffer and average
            results_buffer.append(preds)
            if len(results_buffer) > 10:
                results_buffer.pop(0)
            
            avg_preds = np.mean(results_buffer, axis=0)
            
            idx = np.argmax(avg_preds)
            confidence = avg_preds[idx]
            
            # Display
            label = f"{EMOTION_LABELS[idx]}: {confidence*100:.1f}%"
            color = (0, 255, 0) # Green
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Emotion AI - Stabilized", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()