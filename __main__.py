from __future__ import print_function
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import time

nn = __import__('neural_network')

# Generating models
# model_h, model_g = nn.nn_go()

# Recreating saved models, including weights and optimizer
model_h = keras.models.load_model('C:\\Apps\\PyCharm Community Edition 2019.2.4\\PROJEKTY\Gesture_recognition_using_USB_camera\\model_h_plain_background.h5')
model_g = keras.models.load_model('C:\\Apps\\PyCharm Community Edition 2019.2.4\\PROJEKTY\Gesture_recognition_using_USB_camera\\model_g_plain_background.h5')

gesture_history = [-1] * 10
gesture_history_counter = 0

video = cv2.VideoCapture(0)
check, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

height_len = len(gray)  # Height of input image
width_len = len(gray[0])  # Width of input image
factor = 2  # Factor used to change dimension

frame_out = np.array([[0] * (int(width_len / factor))] * int((height_len / factor)),
                     dtype=np.uint8)  # Table with new dimension

def increase():
    global prediction_text
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if volume.GetMasterVolume() < 0.9:
            volume.SetMasterVolume(volume.GetMasterVolume() + 0.1, None)
    prediction_text = "Volume up: " + str(round(volume.GetMasterVolume()*100)) + "%"
    return "Volume up: " + str(round(volume.GetMasterVolume()*100)) + "%"

def decrease():
    global prediction_text
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if volume.GetMasterVolume() > 0.1:
            volume.SetMasterVolume(volume.GetMasterVolume() - 0.1, None)
    prediction_text = "Volume down: " + str(round(volume.GetMasterVolume() * 100)) + "%"
    return "Volume down: " + str(round(volume.GetMasterVolume()*100)) + "%"

def mute():
    global prediction_text
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        volume.SetMute(not volume.GetMute(), None)
    if volume.GetMute() == 1: prediction_text = "Mute ON"
    else: prediction_text = "Mute OFF"
    return "Mute: " + str(volume.GetMute())

def add_n_check_history(value):
    global gesture_history, gesture_history_counter

    if gesture_history_counter >= len(gesture_history):
        gesture_history_counter = 0

    gesture_history[gesture_history_counter] = value
    gesture_history_counter += 1

    for i in gesture_history:
        if i is not value: return False

    gesture_history = [-1] * 10

    return True


while True:
    sessions = AudioUtilities.GetAllSessions()
    check, frame = video.read()  # Create a frame object
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting to grayscale

    # Refactoring image
    for i in range(0, height_len, factor):
        for j in range(0, width_len, factor):
            frame_out[int((i / factor))][int(j / factor)] = gray[i][j]

    cv2.rectangle(frame_out, (0, 0), (120, 25), 0, -1)
    cv2.rectangle(frame_out, (int(width_len/factor/2)-int(76/factor), int(height_len/factor/2)-int(76/factor)), (int(width_len/factor/2)+int(76/factor), int(height_len/factor/2)+int(76/factor)), 255, 1)
    cv2.rectangle(frame_out, (0, int(height_len/factor)-15), (120, int(height_len/factor)), 0, -1)
    cv2.putText(frame_out, "Press 'q' for quit", (0, int(height_len/factor)-13), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.9, color=255, thickness=1)

    frame_resized = np.array([[0] * 38] * 38, dtype=np.uint8)
    frame_extract = cv2.getRectSubPix(gray, (152, 152), (int(width_len/2), int(height_len/2)))  # Extracting frame
    for i in range(0, 152, 4):
        for j in range(0, 152, 4):
            frame_resized[int(i/4)][int(j/4)] = frame_extract[i][j]
    frame_resized.resize([1, 38, 38, 1])
    frame_resized = tf.cast(frame_resized, tf.float32)
    prediction_result_h = model_h.predict(frame_resized)  # Prediction for hand/no hand

    if prediction_result_h[0][0] >= 0.75:
        prediction_result_g = model_g.predict(frame_resized)  # Prediction for gestures
        if prediction_result_g[0][0] >= 0.7:
            # Fist
            if add_n_check_history(0): decrease()
        elif prediction_result_g[0][1] >= 0.7:
            # Palm
            if add_n_check_history(1): increase()
        elif prediction_result_g[0][2] >= 0.65:
            # Point
            if add_n_check_history(2): mute()
        else:
            prediction_text = "Sth went wrong"
    else:
        add_n_check_history(-1)
        prediction_text = ""

    cv2.putText(frame_out, prediction_text, (0, 18), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.4, color=255,
                thickness=1)

    cv2.imshow("Hello World!", frame_out)  # Show the frame
    key = cv2.waitKey(1)  # For playing

    if key == ord('q'):
        break

# Shutdown the camera
video.release()
cv2.destroyAllWindows()
