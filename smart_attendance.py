import cv2
import os
import numpy as np
import pickle
from datetime import datetime, date

# ========== Paths ==========
dataset_dir = "dataset"
model_path = "model.yml"
names_path = "names.pkl"
attendance_file = "attendance.csv"

# ========== Initialize ==========
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("❌ Failed to load face cascade classifier.")
    exit()

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# ========== Attendance Logic ==========
def mark_attendance(name, is_checkout=False):
    today = date.today().isoformat()
    now = datetime.now().strftime("%H:%M:%S")

    lines = []
    if os.path.exists(attendance_file):
        with open(attendance_file, "r") as f:
            lines = f.read().splitlines()

    section_header = f"===== {today} ====="
    if section_header not in lines:
        lines.append(section_header)
        lines.append("Name,Check-In Time,Check-Out Time")
        lines.append(f"{name},{now},-")
        with open(attendance_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"✅ Check-in recorded for {name} at {now}")
        return True

    idx = lines.index(section_header)
    start = idx + 2
    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].startswith("====="):
            end = i
            break

    found = False
    for i in range(start, end):
        parts = lines[i].split(",")
        if parts[0] == name:
            if is_checkout and parts[2] == "-":
                parts[2] = now
                lines[i] = ",".join(parts)
                with open(attendance_file, "w") as f:
                    f.write("\n".join(lines) + "\n")
                print(f"✅ Check-out recorded for {name} at {now}")
                return True
            elif parts[2] == "-":
                print(f"⏳ {name} already checked in at {parts[1]}. Press 'p' to check out.")
            else:
                print(f"ℹ️ {name} already checked out today at {parts[2]}.")
            found = True
            break

    if not found and not is_checkout:
        lines.insert(end, f"{name},{now},-")
        with open(attendance_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"✅ Check-in recorded for {name} at {now}")
        return True

    return False

# ========== Model Utilities ==========
def train_model():
    data, labels = [], []
    name_to_id = {}
    current_id = 0

    for file in os.listdir(dataset_dir):
        if file.endswith(".jpg"):
            name = os.path.splitext(file)[0].rsplit("_", 1)[0]  # Get name before underscore
            img_path = os.path.join(dataset_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if name not in name_to_id:
                    name_to_id[name] = current_id
                    current_id += 1
                data.append(img)
                labels.append(name_to_id[name])

    if data:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(data, np.array(labels))
        model.save(model_path)
        with open(names_path, "wb") as f:
            pickle.dump({v: k for k, v in name_to_id.items()}, f)
        print(f"✅ Model trained on {len(data)} images.")
    else:
        print("⚠️ No face data found to train the model.")

def load_model():
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read(model_path)
    with open(names_path, "rb") as f:
        id_to_name = pickle.load(f)
    return model, id_to_name

def capture_new_face(name, frame, coords):
    (x, y, w, h) = coords
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    count = len([f for f in os.listdir(dataset_dir) if f.startswith(name)])

    for i in range(5):  # Capture 5 images
        face = gray[y:y+h, x:x+w]
        path = os.path.join(dataset_dir, f"{name}_{count + i + 1}.jpg")
        cv2.imwrite(path, face)
        print(f"📸 Saved face: {path}")
        cv2.waitKey(300)  # Small delay

    train_model()
    print("🔄 Model retrained.")

# ========== First-Time Training ==========
if not os.path.exists(model_path) or not os.path.exists(names_path):
    print("🔄 Model not found. Capture new face to start.")

# ========== Start Camera ==========
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("🎥 Smart Face Attendance started. Press 'q' to quit. Press 's' to save new face. Press 'p' to check out.")

model = None
names = {}
if os.path.exists(model_path) and os.path.exists(names_path):
    model, names = load_model()

marked_names = set()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    key = cv2.waitKey(1) & 0xFF

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]

        if model:
            label, confidence = model.predict(roi)
            name = names.get(label, "Unknown")

            if confidence < 80:
                is_checkout = key == ord('p')

                if name not in marked_names:
                    if mark_attendance(name, is_checkout=False):
                        marked_names.add(name)
                    cv2.putText(frame, f"{name} - Checked In", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif is_checkout:
                    if mark_attendance(name, is_checkout=True):
                        marked_names.remove(name)
                    cv2.putText(frame, f"{name} - Checked Out", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, f"{name} - Press 'p' to Check Out", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                continue

        # Unknown face
        cv2.putText(frame, "New Face - Press 's' to Save", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        if key == ord('s'):
            print("🆕 New face detected. Please type the name in console.")
            new_name = input("Enter name: ").strip()
            if new_name:
                capture_new_face(new_name, frame, (x, y, w, h))
                mark_attendance(new_name, is_checkout=False)
                model, names = load_model()
                marked_names.add(new_name)

    cv2.imshow("Smart Attendance", frame)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
