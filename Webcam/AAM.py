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
import time
import math

from os import listdir
from os.path import isfile, join


def argument_parser():
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
    time.sleep(1)
    return (detector, predictor, vs)


def get_file_names(path):
    exp_names = listdir(path)
    image_names = []
    for p in exp_names:
        image_names.append([join(path, p, f) for f in listdir(join(path, p)) if
                            isfile(join(path, p, f))])
    return image_names


def draw(frame, x0, y0, w, h, landmarks):
    # Cor em BGR
    # cv2.rectangle(frame, (x0, y0), (x0+w, y0+h), (0, 255, 0), 2)

    landmarks_points = []
    for n in range(0, 27):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))

    for i in range(0, len(landmarks_points)-1):
        # cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        if i != 16:
            cv2.line(frame, landmarks_points[i],
                     landmarks_points[i+1], (0, 0, 255), 2)
        else:
            cv2.line(frame, landmarks_points[i],
                     landmarks_points[-1], (0, 0, 255), 2)
        if i == 0:
            cv2.line(frame, landmarks_points[i],
                     landmarks_points[17], (0, 0, 255), 2)


def pad_image(image, d_width, d_height):
    ht, wd, cc = image.shape

    color = (0, 0, 0)
    result = np.full((d_height, d_width, cc), color, dtype=np.uint8)

    xx = (d_width - wd) // 2
    yy = (d_height - ht) // 2

    result[yy:yy+ht, xx:xx+wd] = image
    return result


if __name__ == "__main__":
    i = 0
    j = 0
    image_names = get_file_names('../../Documentos/PAIN/Images/043-jh043/')
    (detector, predictor, webcam) = argument_parser()

    while True and i < len(image_names):
        # frame = vs.read()
        frame = cv2.imread(image_names[i][j])

        j = j+1
        if j == len(image_names[i]):
            j = 0
            i = i+1
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t0 = time.monotonic_ns()
        faces = detector(gray, 0)

        for face in faces:

            shape = predictor(gray, face)

            (x0, y0, w, h) = face_utils.rect_to_bb(face)
            draw(frame, *(x0, y0, w, h), shape)
            shape = face_utils.shape_to_np(shape)

            remap = np.zeros_like(shape)
            out_face = np.zeros_like(frame)
            feature_mask = np.zeros((frame.shape[0], frame.shape[1]))
            remap = cv2.convexHull(shape)
            cv2.fillConvexPoly(feature_mask, remap[0:27], 1)
            feature_mask = feature_mask.astype(np.bool)
            out_face[feature_mask] = frame[feature_mask]

            # Recorta o rosto
            out_face = out_face[y0:y0+h, x0:x0+w]

            out_face = pad_image(out_face, 200, 200)

            cv2.imshow("Frame", out_face)

            t1 = time.monotonic_ns()
            # dt = 1/30 - (t1-t0)/1e9
            # if dt > 0:
            # time.sleep(dt)
            # t1 = time.monotonic_ns()
            print("FPS:", round(1e9/(t1-t0)), end="\r")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    cv2.destroyAllWindows()
    webcam.stop()
