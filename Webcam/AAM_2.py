import face_alignment
from skimage import io
import cv2
import os
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def draw(frame, landmarks):
    # Cor em BGR

    #      print(len(landmarks[0]), landmarks[0][2, :][0])

    for i in range(len(landmarks[0])-1):
        cv2.circle(frame, (landmarks[0][i, :][0],
                           landmarks[0][i, :][1]), 2, (0, 0, 255), -1)
#          cv2.line(frame, (landmarks[0][i, :][0], landmarks[0][i, :][1]),
#                   (landmarks[0][i+1, :][0], landmarks[0][i+1, :][1]),
#                   (0, 0, 255), 2)


fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, device='cpu')
#  fa = face_alignment.FaceAlignment(
#      face_alignment.LandmarksType._2D, flip_input=False)

t0 = time.monotonic_ns()
input_im = cv2.imread(
    '../../Documentos/PAIN/Images/047-jl047/jl047t2afaff/jl047t2afaff036.png')
preds = fa.get_landmarks(input_im)
t1 = time.monotonic_ns()

print((t1-t0)/1e9)

draw(input_im, preds)
cv2.imshow("Teste", input_im)

cv2.waitKey()
cv2.destroyAllWindows()
