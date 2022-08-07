"""Face Area detection.

Using opencv to detect the face and determine the area.
"""

import cv2 as cv
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)

df = pd.read_excel("data/frame.xls")


def create_face_area_data():

    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

    df = pd.read_excel("data/frame.xls")

    face_area = []
    for img in df["image"]:
        img_path = "./data/images/" + str(img) + ".jpg"

        img_object = cv.imread(img_path)
        gray = cv.cvtColor(img_object, cv.COLOR_BGR2GRAY)

        # problems #1 - trouble determining if 'multiple faces' are detected
        # problems #2 - trouble if no faces are detected
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (_, _, w, h) in faces:
            face_area.append(100 * w * h / (1920 * 1080))
            break

    # logging.debug(all_face_area)
    df["face area (pixel)"] = face_area

    return df
