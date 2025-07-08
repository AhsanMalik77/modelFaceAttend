import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
import datetime
import os

# --- Configuration ---
DISPLAY_WINDOW = True  # Set to False if running on headless server (no GUI)

# --- Load classifier and labels ---
classifier_path = 'face_classifier2.pkl'
# labels_path = 'student_labels1.npy'

if not os.path.exists(classifier_path):
    print("❌ Classifier or label file missing.")
    exit()

classifier = joblib.load(classifier_path)
# labels = np.load(labels_path, allow_pickle=True)

# --- Device configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# --- Initialize face detection and embedding models ---
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

seen_names = set()

# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Real-time face recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)

    # Face detection
    boxes, probs = mtcnn.detect(img)
    if boxes is not None:
        faces = mtcnn.extract(img, boxes, save_path=None)
        for i, (face, prob) in enumerate(zip(faces, probs)):
            if prob is not None and prob > 0.90:
                with torch.no_grad():
                    embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
                    pred = classifier.predict(embedding)
                    pred_prob = classifier.predict_proba(embedding).max()

                x1, y1, x2, y2 = map(int, boxes[i])

                if pred_prob > 0.90:
                    name = pred[0]
                    label = f"{name} ({pred_prob * 100:.1f}%)"
                    color = (0, 255, 0)

                    if name not in seen_names:
                        seen_names.add(name)
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open("attendance_log.txt", "a") as f:
                            f.write(f"{timestamp} - {name} - Confidence: {pred_prob:.2f}\n")
                        print(f"📝 Attendance marked for: {name}")
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the result
    if DISPLAY_WINDOW:
        try:
            cv2.imshow("🎯 Real-Time Face Recognition", frame)
        except cv2.error as e:
            print("⚠️ Warning: Cannot display window. GUI might not be supported.")
            DISPLAY_WINDOW = False
    else:
        # Save frame for debug (optional)
        cv2.imwrite("debug_frame.jpg", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("🛑 Real-time face recognition stopped.")
