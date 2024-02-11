import numpy as np
from PIL import Image
import os,cv2

def train_classifier(data1_dir):
    path = [os.path.join(data1_dir, f)for f in os.listdir(data1_dir)]
    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img,'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier1.yml")

train_classifier("data1")



