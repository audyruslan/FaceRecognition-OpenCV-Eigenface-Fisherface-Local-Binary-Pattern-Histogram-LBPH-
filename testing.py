import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

dataset_folder = "dataset/"

names = []
images = []
for folder in os.listdir(dataset_folder):
    # limit only 70 face per class
    for name in os.listdir(os.path.join(dataset_folder, folder))[:70]:
        img = cv2.imread(os.path.join(dataset_folder + folder, name))
        images.append(img)
        names.append(folder)


labels = np.unique(names)

face_cascade = cv2.CascadeClassifier(
    'haarcascades/haarcascade_frontalface_default.xml')


def detect_face(img, idx):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    try:
        x, y, w, h = faces[0]

        img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (100, 100))
    except:
        print("Face not found in image index", i)
        img = None
    return img


croped_images = []
for i, img in enumerate(images):
    img = detect_face(img, i)
    if img is not None:
        croped_images.append(img)
    else:
        del names[i]

model = cv2.face.FisherFaceRecognizer_create()
model.read("model/fisher_model.yml")


# Testing 1 Gambar
# path = "test/Colin_Powell_0116.jpg"

# img = cv2.imread(path)
# img = detect_face(img, 0)

# idx, confidence = model.predict(img)

# print("Found: ", labels[idx])
# print("Confidence: ", confidence)

# plt.figure()
# plt.imshow(img, cmap="gray")
# plt.axis(False)
# plt.show()


# Testing Gambar 1 Folder
test_folder = "test/"
actual_names = []
predicted_names = []
confidences = []
for filename in os.listdir(test_folder):
    if filename.find(".jpg") > -1:
        path = os.path.join(test_folder, filename)

        img = cv2.imread(path)
        img = detect_face(img, 0)

        idx, confidence = model.predict(img)

        # get label from filename (remove 9 last char)
        actual_names.append(np.where(filename[:-9] == labels)[0][0])
        predicted_names.append(idx)
        confidences.append(confidence)

        print("Actual \t: ", filename)
        print("Predic \t: ", labels[idx])
        print("Confidence \t: ", confidence)

        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.axis(False)
        plt.show()
