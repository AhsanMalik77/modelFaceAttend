from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib
from flask_cors import CORS
import datetime
import os

app = Flask(__name__)
CORS(app)

# Load trained classifier (ensure it's uploaded on Railway or downloaded at runtime)
classifier = joblib.load('face_classifier2.pkl')

# Setup device and face recognition models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Seen names (reset each server session)
seen_names = set()

@app.route('/health')
def health():
    return jsonify({"status": "Face Recognition API is running"}), 200

def recognize_faces(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes, probs = mtcnn.detect(img)

    results = []

    if boxes is not None:
        faces = mtcnn.extract(img, boxes, save_path=None)
        for i, (face, prob) in enumerate(zip(faces, probs)):
            if prob and prob > 0.80:
                with torch.no_grad():
                    embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
                    pred = classifier.predict(embedding)
                    pred_prob = classifier.predict_proba(embedding).max()

                x1, y1, x2, y2 = map(int, boxes[i])
                name = pred[0] if pred_prob > 0.80 else "Unknown"

                # Optional: Add DB logging here instead of file
                if name != "Unknown" and name not in seen_names:
                    seen_names.add(name)
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"{timestamp} - {name} - Confidence: {pred_prob:.2f}")

                results.append({
                    "name": name,
                    "confidence": float(pred_prob),
                    "box": [x1, y1, x2, y2]
                })
    else:
        print("❌ No face detected.")

    return results

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    try:
        results = recognize_faces(img)
        return jsonify({"results": results}), 200
    except Exception as e:
        print("❌ Error during recognition:", str(e))
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # use Railway-provided port
    app.run(host='0.0.0.0', port=port)
