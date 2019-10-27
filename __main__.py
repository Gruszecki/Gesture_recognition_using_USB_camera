from __future__ import print_function
import cv2
import numpy as np
from tensorflow import keras
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import time

nn = __import__('neural_network')

# Generating models
model_h, model_g = nn.nn_go()

# Recreating saved models, including weights and optimizer
# model_h = keras.models.load_model('F:\\PyCharm 5.0.4\\PROJEKTY\\ExternalCamera\\model_h_three_gest.h5')
# model_g = keras.models.load_model('F:\\PyCharm 5.0.4\\PROJEKTY\\ExternalCamera\\model_g_three_gest.h5')

# Create an object.
video = cv2.VideoCapture(0)

check, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

height_len = len(gray)  # Height of input image
width_len = len(gray[0])  # Width of input image
factor = 4  # Factor used to change dimension

frame_out = np.array([[0] * (int(width_len / factor))] * int((height_len / factor)),
                     dtype=np.uint8)  # Table with new dimension

def increase():
    print("Volume increasing.")
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if volume.GetMasterVolume() < 0.9:
            volume.SetMasterVolume(volume.GetMasterVolume() + 0.1, None)

def decrease():
    print("Volume decreasing.")
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        if volume.GetMasterVolume() > 0.1:
            volume.SetMasterVolume(volume.GetMasterVolume() - 0.1, None)

def mute():
    print("Mute ON/OFF")
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        volume.SetMute(not volume.GetMute(), None)


while True:
    sessions = AudioUtilities.GetAllSessions()
    check, frame = video.read()  # Create a frame object
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting to grayscale

    # Refactoring image
    for i in range(0, height_len, factor):
        for j in range(0, width_len, factor):
            frame_out[int((i / factor))][int(j / factor)] = gray[i][j]

    cv2.rectangle(frame_out, (0, 0), (60, 15), 0, -1)
    cv2.rectangle(frame_out, (60, 40), (100, 80), 255, 1)

    frame_extract = cv2.getRectSubPix(frame_out, (38, 38), (80, 60))  # Extracting frame
    frame_extract.resize([1, 38, 38, 1])
    prediction_result_h = model_h.predict(frame_extract)  # Prediction for hand/no hand

    if prediction_result_h[0][0] >= 0.75:
        prediction_result_g = model_g.predict(frame_extract)  # Prediction for gestures
        if prediction_result_g[0][0] >= 0.7:
            prediction_text = "Fist"
            decrease()
        elif prediction_result_g[0][1] >= 0.7:
            prediction_text = "Palm"
            mute()
        elif prediction_result_g[0][2] >= 0.7:
            prediction_text = "Point"
            increase()
        else:
            prediction_text = "Sth went wrong"
        time.sleep(0.5)
    else:
        prediction_text = "No hand"

    cv2.putText(frame_out, prediction_text, (0, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.9, color=255,
                thickness=1)

    cv2.imshow("Hello World!", frame_out)  # Show the frame
    key = cv2.waitKey(1)  # For playing

    if key == ord('q'):
        break

# Shutdown the camera
video.release()
cv2.destroyAllWindows()
