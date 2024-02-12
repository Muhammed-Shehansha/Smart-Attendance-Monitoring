import os
from os import listdir
from PIL import Image as Img
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from keras.models import load_model
import numpy as np
import csv

import pickle
import cv2

HaarCascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))
MyFaceNet = load_model("facenet_keras.h5")

folder = 'ImageAttendance/'
database = {}

for filename in listdir(folder):

    path = folder + filename
    gbr1 = cv2.imread(path)

    wajah = HaarCascade.detectMultiScale(gbr1, 1.1, 4)

    if len(wajah) > 0:
        x1, y1, width, height = wajah[0]
    else:
        x1, y1, width, height = 1, 1, 10, 10

    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    gbr = Img.fromarray(gbr)  # konversi dari OpenCV ke PIL
    gbr_array = asarray(gbr)

    face = gbr_array[y1:y2, x1:x2]

    face = Img.fromarray(face)
    face = face.resize((160, 160))
    face = asarray(face)

    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std

    face = expand_dims(face, axis=0)
    signature = MyFaceNet.predict(face)

    database[os.path.splitext(filename)[0]] = signature
    # print(database)

myfile = open("data.pkl", "wb")
pickle.dump(database, myfile)
myfile.close()

myfile = open("data.pkl", "rb")
database = pickle.load(myfile)
myfile.close()


fields = ['Name', 'Time']
with open("Attendance.csv", 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)



