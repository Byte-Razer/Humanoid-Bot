

from pathlib import Path
import numpy as np
import cv2
import pyttsx3
import sys


BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / "data"
assets_path = BASE_DIR / "assets"

modelFile = str(assets_path / "res10_300x300_ssd_iter_140000.caffemodel")
configFile = str(assets_path / "deploy.prototxt")


def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())
    

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]



net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


face_data = []
labels = []
names = {}
class_id = 0

if not dataset_path.exists():
    print("'data' folder not found. Please run train.py first.")
    sys.exit()

for file in dataset_path.glob("*.npy"):
    names[class_id] = file.stem
    print(" Loaded:", file.name)

    data_item = np.load(file)
    data_item = data_item.reshape(data_item.shape[0], -1)
    face_data.append(data_item)

    target = class_id * np.ones((data_item.shape[0],))
    labels.append(target)
    class_id += 1

if not face_data:
    print("No training data found in ./data/. Please collect faces first.")
    sys.exit()

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

print("\n Training data loaded successfully!")
print("   Face dataset shape:", face_dataset.shape)
print("   Face labels shape:", face_labels.shape)


engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
spoken_names = set()


USE_LBPH = True   # Set to False to disable LBPH

if USE_LBPH:
    print("\nInitializing LBPH recognizer (tuned parameters)...")
    lbph = cv2.face.LBPHFaceRecognizer_create(
        radius=1,        # Slightly larger radius for lighting robustness
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=70.0  # Adjust for your dataset
    )
    # Prepare data for LBPH training (grayscale images)
    faces, face_ids = [], []
    for file in dataset_path.glob("*.npy"):
        current_id = [key for key, name in names.items() if name == file.stem][0]
        data = np.load(file)
        for img in data:
            if len(img.shape) == 3:  # Color image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:  # Already grayscale
                gray = img

            gray = cv2.equalizeHist(gray)
            faces.append(gray)
            face_ids.append(current_id)
    if faces:
        lbph.train(faces, np.array(face_ids))
        print("LBPH training complete!")


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam. Try changing the camera index.")
    sys.exit()

print("\nPress 'r' to reset spoken names, 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face_section = frame[y1:y2, x1:x2]
            if face_section.size == 0:
                continue

            # -----------------------------
            # Normalize face
            # -----------------------------
            face_section = cv2.cvtColor(face_section, cv2.COLOR_BGR2GRAY)
            face_section = cv2.equalizeHist(face_section)
            face_section = cv2.resize(face_section, (128, 128))

            # -----------------------------
            # Predict using LBPH or KNN
            # -----------------------------
            if USE_LBPH:
                label, confidence_value = lbph.predict(face_section)
                if label >= 0 and confidence_value < 150:
                    pred_name = names[label]
                else:
                    pred_name = names.get(label, "Unknown")
            else:
                out = knn(trainset, face_section.flatten())
                pred_name = names[int(out)]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"{pred_name} ({confidence * 100:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

            if pred_name not in spoken_names and pred_name != "Unknown":
                engine.say(f"Hi {pred_name}. Welcome to Utpal Shangvi Global School!")
                engine.runAndWait()
                spoken_names.add(pred_name)

    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        spoken_names.clear()
        print(" Reset spoken names.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

