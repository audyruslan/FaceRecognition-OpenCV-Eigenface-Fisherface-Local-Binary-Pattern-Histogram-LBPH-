import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_dataset(images_class, label):
    # show data for 1 class
    plt.figure(figsize=(14, 5))
    k = 0
    for i in range(1, 6):
        plt.subplot(1, 5, i)
        try:
            plt.imshow(images_class[k][:, :, ::-1])
        except:
            plt.imshow(images_class[k], cmap='gray')
        plt.title(label)
        plt.axis('off')
        plt.tight_layout()
        k += 1
    plt.show()


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

# print(labels)

# for label in labels:

#     ids = np.where(label == np.array(names))[0]
#     images_class = images[ids[0]: ids[-1] + 1]
#     show_dataset(images_class, label)

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


# for label in labels:

#     ids = np.where(label == np.array(names))[0]
#     # select croped images for each class
#     images_class = croped_images[ids[0]: ids[-1] + 1]
#     show_dataset(images_class, label)

# print(names)
# print(labels)

name_vec = np.array([np.where(name == labels)[0][0] for name in names])
# print(name_vec)


# EigenFaceRecognizer Model
# model = cv2.face.EigenFaceRecognizer_create()
# model.train(croped_images, name_vec)
# model.save("model/eigen_model.yml")

# FisherFaceRecognizer Model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(croped_images, name_vec)
model.save("model/fisher_model.yml")

# LBPHFaceRecognizer Model
# model = cv2.face.FisherFaceRecognizer_create()
# model.train(croped_images, name_vec)
# model.save("model/lbph_model.yml")
