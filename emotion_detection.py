import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from deepface import DeepFace

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "emotion_model.h5")  # adjust if needed

# ================= LOAD CNN MODEL =================
cnn_model = load_model(model_path, compile=False)

emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# ================= SETTINGS =================
ANALYZE_EVERY = 6   # DeepFace runs every N frames
prediction_history = []
frame_count = 0

# ================= FACE DETECTOR =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

print("🚀 Hybrid Emotion Detection Started (Press Q to exit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔥 Resize for performance
    frame_small = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    # 🔥 Improve contrast
    gray = cv2.equalizeHist(gray)

    # 🔥 Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(60, 60)
    )

    frame_count += 1

    for (x, y, w, h) in faces:

        # ================= CNN (FAST) =================
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))   # 🔥 FIX SIZE
        roi = roi / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))  # 🔥 FIX SHAPE
        cnn_pred = cnn_model.predict(roi, verbose=0)[0]

        # ================= DEEPFACE (ACCURATE) =================
        if frame_count % ANALYZE_EVERY == 0:
            try:
                face_img = frame_small[y:y+h, x:x+w]

                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    detector_backend='opencv',  # fast backend
                    enforce_detection=False
                )

                if isinstance(result, list):
                    result = result[0]

                df_emotions = result["emotion"]

                deepface_pred = np.array([
                    df_emotions[label] for label in emotion_labels
                ]) / 100

            except Exception as e:
                print("DeepFace error:", e)
                deepface_pred = cnn_pred  # fallback
        else:
            deepface_pred = cnn_pred

        # ================= HYBRID =================
        final_pred = (0.6 * cnn_pred) + (0.4 * deepface_pred)

        # ================= SMOOTHING =================
        prediction_history.append(final_pred)
        if len(prediction_history) > 10:
            prediction_history.pop(0)

        avg_pred = np.mean(prediction_history, axis=0)

        max_index = np.argmax(avg_pred)
        confidence = avg_pred[max_index] * 100
        emotion = emotion_labels[max_index].capitalize()

        # ================= FILTER =================
        if confidence < 35:
            continue

        label = f"{emotion} ({confidence:.1f}%)"

        # ================= DRAW =================
        cv2.rectangle(frame_small, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame_small,
            label,
            (x, y-10 if y > 20 else y+h+20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("🔥 Hybrid Emotion Detector", frame_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()