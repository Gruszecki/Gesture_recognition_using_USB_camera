from __future__ import print_function
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import csv

nn = __import__('neural_network')

import_models_flag = 1      # 1 for import, 0 for generate
sil_flag = 1                # 1 for sil, 0 for live
sil_dir = "output_final.mp4"

if import_models_flag:
    # Recreating saved models, including weights and optimizer
    model_h = keras.models.load_model(
        'C:\\Apps\\PyCharm Community Edition 2019.3.3\\PROJEKTY\\Gesture_recognition_using_USB_camera\\model_h_monochrome.h5')
    model_g = keras.models.load_model(
        'C:\\Apps\\PyCharm Community Edition 2019.3.3\\PROJEKTY\\Gesture_recognition_using_USB_camera\\model_g_monochrome.h5')
else:
    # Generating models
    model_h, model_g = nn.nn_go()

# Gesture history init
gesture_history = [-1] * 20
gesture_history_counter = 0
predicted_handpose = ""
predicted_gesture = ""

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

    gesture_history = [-1] * len(gesture_history)

    return True

def video_source(sil_flag):
    if sil_flag: return cv2.VideoCapture(sil_dir)
    else: return cv2.VideoCapture(0)

def check_gesture(frame_resized):
    global prediction_text, predicted_handpose, predicted_gesture
    prediction_result_h = model_h.predict(frame_resized)  # Prediction for hand/no hand
    if prediction_result_h[0][0] >= 0.8:
        prediction_result_g = model_g.predict(frame_resized)  # Prediction for gestures
        if prediction_result_g[0][0] >= 0.5:
            # Fist
            predicted_handpose = 'Fist'
            if add_n_check_history(0):
                predicted_gesture = 'Fist'
                decrease()
        elif prediction_result_g[0][1] >= 0.5:
            # Palm
            predicted_handpose = 'Palm'
            if add_n_check_history(1):
                predicted_gesture = 'Palm'
                increase()
        elif prediction_result_g[0][2] >= 0.5:
            # Point
            predicted_handpose = 'One finger'
            if add_n_check_history(2):
                predicted_gesture = 'One finger'
                mute()
        else:
            prediction_text = "Sth went wrong"
    else:
        add_n_check_history(-1)
        prediction_text = ""

def refactor_image(graycome):
    for i in range(0, height_len, factor):
        for j in range(0, width_len, factor):
            frame_out[int((i / factor))][int(j / factor)] = graycome[i][j]
    return frame_out

def resize_image_38(graycome):
    frame_holder = np.array([[0]*38]*38, dtype=np.uint8)
    frame_extract = cv2.getRectSubPix(graycome, (152, 152), (int(width_len/2), int(height_len/2)))  # Extracting frame
    for i in range(0, 152, 4):
        for j in range(0, 152, 4):
            frame_holder[int(i/4)][int(j/4)] = frame_extract[i][j]
    frame_holder.resize([1, 38, 38, 1])
    frame_holder = tf.cast(frame_holder, tf.float32)
    return frame_holder


# Stream creating
video = video_source(sil_flag)
check, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Stream properties
height_len = len(gray)  # Height of input image
width_len = len(gray[0])  # Width of input image
factor = 2  # Factor used to change dimension

frame_out = np.array([[0] * (int(width_len / factor))] * int((height_len / factor)),
                     dtype=np.uint8)  # Table with new dimension

prediction_text = ""

if sil_flag:
    stream_holder = []
    while check:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_out = refactor_image(gray)
        stream_holder.append([frame_out, gray])
        check, frame = video.read()

    with open('sil_result.csv', 'w', newline='') as file:
        filewriter = csv.writer(file, delimiter='\t')
        filewriter.writerow(['Numer klatki', 'Handpose', 'Gest (SIL)', 'Gest (etykieta)'])
        counter = 1
        for frame_holder in stream_holder:
            sessions = AudioUtilities.GetAllSessions()

            cv2.rectangle(frame_holder[1], (0, 0), (120, 25), 0, -1)
            cv2.rectangle(frame_holder[1], (int(width_len/2)-int(height_len/6), int(height_len/3)), (int(width_len/2)+int(height_len/6), int(height_len/3)*2), 255, 1)
            cv2.rectangle(frame_holder[1], (0, int(height_len)-15), (120, int(height_len)), 0, -1)
            cv2.putText(frame_holder[1], "Press 'q' for quit", (0, int(height_len)-13), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.9, color=255, thickness=1)

            frame_resized = resize_image_38(frame_holder[1])
            check_gesture(frame_resized)
            filewriter.writerow([counter, predicted_handpose, predicted_gesture])
            predicted_handpose = ''
            predicted_gesture = ''
            counter += 1

            cv2.putText(frame_holder[1], prediction_text, (0, 18), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.4, color=255,
                        thickness=1)

            cv2.imshow("Hello World!", frame_holder[1])
            key = cv2.waitKey(1)  # For playing

            if key == ord('q'):
                break
else:
    while True:
        sessions = AudioUtilities.GetAllSessions()
        check, frame = video.read()  # Create a frame object

        # Refactoring image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_out = refactor_image(gray)

        cv2.rectangle(frame_out, (0, 0), (120, 25), 0, -1)
        cv2.rectangle(frame_out,
                      (int(width_len / factor / 2) - int(76 / factor), int(height_len / factor / 2) - int(76 / factor)),
                      (int(width_len / factor / 2) + int(76 / factor), int(height_len / factor / 2) + int(76 / factor)),
                      255, 1)
        cv2.rectangle(frame_out, (0, int(height_len / factor) - 15), (120, int(height_len / factor)), 0, -1)
        cv2.putText(frame_out, "Press 'q' for quit", (0, int(height_len / factor) - 13),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.9, color=255, thickness=1)

        frame_resized = resize_image_38(gray)
        check_gesture(frame_resized)

        cv2.putText(frame_out, prediction_text, (0, 18), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.4, color=255,
                    thickness=1)

        cv2.imshow("Hello World!", frame_out)  # Show the frame
        key = cv2.waitKey(1)  # For playing

        if key == ord('q'):
            break

# Shutdown the camera
video.release()
cv2.destroyAllWindows()
