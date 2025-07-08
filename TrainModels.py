import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.svm import SVC
import joblib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

data_path = 'Data'
embedding_list = []
name_list = []

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

for student in os.listdir(data_path):
    student_path = os.path.join(data_path, student)
    if not os.path.isdir(student_path): continue

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)

        if face is not None:
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
                embedding_list.append(emb[0])  # FIXED HERE
                name_list.append(student)

X = np.array(embedding_list)
y = np.array(name_list)

classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y)

joblib.dump(classifier, 'face_classifier.pkl')
np.save('student_labels.npy', y)

print("✅ Model trained and saved successfully!")
