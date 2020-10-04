
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:28:50 2020

@author: sid
"""

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import argparse

def preprocess(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image/255.0
    return image

def change(i):
    return (1-i)

def output(i):
    i *= 100
    if i<50:
        print("NEGATIVE")
        print("Confidence Score - "+str(100-i)+"%")
    else:
        print("POSITIVE")
        print("Confidence Score - "+str(i)+"%")


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--imagepath", default="\image.jpg",
	help="path to input image of x-ray")
ap.add_argument("-m", "--modelpath", default="\model.h5")
args = vars(ap.parse_args())

imagepath = args["imagepath"]
modelpath = args["modelpath"]

model = load_model(modelpath)
image = preprocess(imagepath)

data = []
data.append(image)
data = np.array(data)

predict = change(model.predict(data))
output(predict[0][0])
