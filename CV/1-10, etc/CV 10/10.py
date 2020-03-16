import cv2
import os
import numpy as np
import math

train_path = '10/train'
person_names = os.listdir(train_path)

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_list = []
class_list = []
# 0 -> felix
# 1 -> jonas
# 2 -> nayeon
for idx, person_name in enumerate(person_names): # urut sesuai abjad
    full_person_path = train_path + '/' + person_name

    for image_path in os.listdir(full_person_path):
        full_image_path = full_person_path + '/' + image_path
        img_bgr = cv2.imread(full_image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # detect face 
        detected_faces = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5) # scale factor = merescale ukuran image jadi kecil
        if (len(detected_faces) == 0):
            continue       

        for face_rect in detected_faces:
            x, y, w, h = face_rect
            curr_face = img_gray[y:y+h, x:x+w]
            
            face_list.append(curr_face)
            class_list.append(idx)

            # make rectangle
            # cv2.rectangle(
            #     img_bgr, 
            #     (x, y), 
            #     (x + w, y + h), 
            #     (0, 255, 0), # warna kotak
            #     2 # ketebalan garis kotak
            # )
            # cv2.imshow('1', img_bgr)
            # cv2.waitKey(0)
            # exit()

# recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(face_list, np.array(class_list))

test_path = '10/test'

for image_path in os.listdir(test_path):
    full_image_path = test_path + '/' + image_path
    img_bgr = cv2.imread(full_image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    detected_faces = cascade.detectMultiScale(
        img_gray, 
        scaleFactor=1.2, 
        minNeighbors=5
    )
    if(len(detected_faces) == 0):
        continue

    for rect_face in detected_faces:
        x, y, w, h = rect_face
        curr_face = img_gray[y:y+h, x:x+w]

        result, confidence = recognizer.predict(curr_face) # terima 2 parameter : jawaban dan confidence level
        confidence = math.floor(confidence * 100) / 100

        cv2.rectangle(
            img_bgr, 
            (x, y), 
            (x + w, y + h), 
            (0, 255, 0), # warna kotak
            2 # ketebalan garis kotak
        )

        text = person_names[result] + ' ' + str(confidence)
        cv2.putText(
            img_bgr,
            text,
            (x, y),
            cv2.FONT_HERSHEY_PLAIN,
            1.5,
            (0, 255, 0),
            1
        )
        cv2.imshow('1', img_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print(person_names[result] + ' ' + str(confidence))