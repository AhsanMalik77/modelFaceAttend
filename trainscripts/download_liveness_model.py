import requests
from tqdm import tqdm

url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat"
filename = "../shape_predictor_68_face_landmarks.dat"

print("Downloading Dlib shape predictor (this may take a while)...")

response = requests.get(url, stream=True)
total = int(response.headers.get('content-length', 0))

with open(filename, 'wb') as file, tqdm(
    desc=filename,
    total=total,
    unit='B',
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for data in response.iter_content(chunk_size=1024):
        size = file.write(data)
        bar.update(size)

print("Download complete.")
