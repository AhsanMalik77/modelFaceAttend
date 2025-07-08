import threading
import subprocess

def run_api1():
    subprocess.run([r"C:\Users\Malik Moin\PycharmProjects\projectPDc\.venv\Scripts\python.exe", "face_recognition_api.py"])

def run_api2():
    subprocess.run([r"C:\Users\Malik Moin\PycharmProjects\projectPDc\.venv\Scripts\python.exe", "train_api.py"])

t1 = threading.Thread(target=run_api1)
t2 = threading.Thread(target=run_api2)

t1.start()
t2.start()

t1.join()
t2.join()
