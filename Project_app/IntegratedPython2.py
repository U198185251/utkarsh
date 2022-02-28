import serial  # Serial imported for Serial communication
import time  # Required to use delay functions
import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from statistics import mode

# start the connection to arduino
ArduinoUnoSerial = serial.Serial('/dev/ttyACM0', 38400)
print(ArduinoUnoSerial.readline())
time.sleep(1)
print("Python is ready")
# load model
model = model_from_json(open("my_model_62.json", "r").read())
# load weights
model.load_weights('weights_62.h5')

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_haar_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
collect = []
while len(collect) < 200:
    ret, test_img = cap.read()
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(gray_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
        predicted_emotion = emotions[max_index]
        # print(predicted_emotion)
        # time.sleep(0.5)
        # var = "0"

        if predicted_emotion == "angry":
            collect.append('1')
        elif predicted_emotion == "disgust":
            collect.append('2')
        elif predicted_emotion == "fear":
            collect.append('3')
        elif predicted_emotion == "happy":
            collect.append('4')
        elif predicted_emotion == "neutral":
            collect.append('5')
        elif predicted_emotion == "sad":
            collect.append('6')
        else:
            collect.append('7')
        # else:
        #     # var = "NA"
        #     ArduinoUnoSerial.write(b'1000')
        #     print(ArduinoUnoSerial.write(b'1000'))
        #
        #     # print(var.encode())

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # time.sleep(2)

    # ArduinoUnoSerial.write(b'Emo')
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    #emo = mode(collect)
    #ArduinoUnoSerial.write(emo)

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
emo = bytes(mode(collect), encoding='utf8')
print(emo)
#print(type(mode(collect)))
#print(type(bytes(emo, encoding='utf8')))
ArduinoUnoSerial.write(emo)
#ArduinoUnoSerial.write()