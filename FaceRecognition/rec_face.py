#!/usr/bin/python3.8
import face_recognition
import cv2
import numpy as np
import os
import glob


class FaceRec(object):
    def __init__(self):
        pass

    def train_images(self):
        self.faces_encodings = []
        self.faces_names = []
        cur_dir = os.getcwd()
        path = os.path.join(cur_dir, 'faces/')

        # Pega os nomes de todos os arquivos na pasta especificada
        files = [f for f in glob.glob(path + '*.jpg')]
        n_files = len(files)

        names = files.copy()

        # Realiza o treinamento
        for i in range(n_files):
            globals()['image_{}'.format(i)] = \
                face_recognition.load_image_file(files[i])
            globals()['image_encoding_{}'.format(i)] = \
                face_recognition.face_encodings(
                    globals()['image_{}'.format(i)])[0]
            self.faces_encodings.append(
                globals()['image_encoding_{}'.format(i)])

            # Nomes das faces
            names[i] = names[i].replace(path, "")
            names[i] = names[i].replace(".jpg", "")
            self.faces_names.append(names[i])
        print(self.faces_names)

    def use_recognition(self):
        face_locations = []
        frame_encodings = []
        frame_names = []
        process_this_frame = True

        video_cap = cv2.VideoCapture(0)

        while True:
            _, frame = video_cap.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = small_frame[:, :, ::-1]

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small)
                frame_encodings = face_recognition.face_encodings(rgb_small,
                                                                  face_locations)

                frame_names = []
                for face_encoding in frame_encodings:
                    matches = face_recognition.compare_faces(self.faces_encodings,
                                                             face_encoding)
                    name = "Desconhecido"

                    face_distances = face_recognition.face_distance(self.faces_encodings,
                                                                    face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.faces_names[best_match_index]

                    frame_names.append(name)
            process_this_frame = not process_this_frame

            frame = self.show_result(frame, face_locations, frame_names)
# Display the resulting image
            try:
                cv2.imshow('Video', frame)
            except:
                pass
# Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def show_result(self, frame, face_locations, face_names):
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
# Input text label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35),
                          (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        font, 1.0, (255, 255, 255), 1)
            return frame


if __name__ == "__main__":
    recognition = FaceRec()
    recognition.train_images()
    recognition.use_recognition()
