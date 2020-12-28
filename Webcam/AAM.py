#!/bin/python3
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np


# Argument parsing
ap = argparse.ArgumentParser()

ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=0,
                help="wheter or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

print("[INFO] carregando landmark predictor")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
# cap = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    # frame = cap.read()
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    for face in faces:

        shape = predictor(gray, face)

        landmarks_points = []
        for n in range(0, 27):
            x = shape.part(n).x
            y = shape.part(n).y
            landmarks_points.append((x, y))

        shape = face_utils.shape_to_np(shape)

        # for i in range(0, len(landmarks_points)-1):
        # # cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        # if i != 16:
        # cv2.line(frame, landmarks_points[i],
        # landmarks_points[i+1], (0, 0, 255), 2)
        # else:
        # cv2.line(frame, landmarks_points[i],
        # landmarks_points[-1], (0, 0, 255), 2)
        # if i == 0:
        # cv2.line(frame, landmarks_points[i],
        # landmarks_points[17], (0, 0, 255), 2)

        remap = np.zeros_like(shape)
        out_face = np.zeros_like(frame)
        feature_mask = np.zeros((frame.shape[0], frame.shape[1]))

        remap = cv2.convexHull(shape)
        cv2.fillConvexPoly(feature_mask, remap[0:27], 1)
        feature_mask = feature_mask.astype(np.bool)
        out_face[feature_mask] = frame[feature_mask]

        cv2.imshow("Frame", out_face)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
cv2.destroyAllWindows()
# cap.release()
vs.stop()
