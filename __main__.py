import cv2
import numpy as np
from tensorflow import keras
nn = __import__('neural_network')

# Generating models
# model_h, model_g = nn.nn_go()

# Recreating saved models, including weights and optimizer
model_h = keras.models.load_model('F:\\PyCharm 5.0.4\\PROJEKTY\\NeutralNetwork\\model_h.h5')
model_g = keras.models.load_model('F:\\PyCharm 5.0.4\\PROJEKTY\\NeutralNetwork\\model_g.h5')

# Create an object.
video = cv2.VideoCapture(0)

check, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

height_len = len(gray)      # Height of input image
width_len = len(gray[0])    # Width of input image
factor = 4                  # Factor used to change dimension
pixels_to_check = [[80, 60], [80, 79], [61, 79], [99, 79], [61, 60], [99, 60], [80, 41], [61, 41], [99, 41]]
bright_pixel = [80, 60]     # Bright pixel in default ROI (one of pixels_to_check if any)
lowest_pixel = [80, 79]     # The lowest bright pixel in hand
hand_present_flag = False
hand_brightnes = 160

frame_out = np.array([[0]*(int(width_len/factor))]*int((height_len/factor)), dtype=np.uint8)    # Table with new dimension

def find_lowest(cur_pixel):
    # pixel[1] is x (down)
    global hand_present_flag, lowest_pixel
    step = 3
    if frame_out[cur_pixel[1], cur_pixel[0]] > hand_brightnes:
        if cur_pixel[1] >= 119 - step:
            lowest_pixel[1] = cur_pixel[1]
            lowest_pixel[0] = cur_pixel[0]
        elif cur_pixel[0] >= 159 - 19 or cur_pixel[0] <= 0 + 19:
            cur_pixel[0] = 80
        elif frame_out[cur_pixel[1] + step, cur_pixel[0]] >= hand_brightnes:
            cur_pixel[1] = cur_pixel[1] + step
        elif frame_out[cur_pixel[1] + step, cur_pixel[0] + step] >= hand_brightnes:
            cur_pixel[1] = cur_pixel[1] + step
            cur_pixel[0] = cur_pixel[0] + step
        elif frame_out[cur_pixel[1] + step, cur_pixel[0] - step] >= hand_brightnes:
            cur_pixel[1] = cur_pixel[1] + step
            cur_pixel[0] = cur_pixel[0] - step
        else:
            lowest_pixel[1] = cur_pixel[1]
            lowest_pixel[0] = cur_pixel[0]
    else:
        if cur_pixel[1]-step*2 < 20:
            cur_pixel[1] = 20 + step*2
        if frame_out[cur_pixel[1] - step*2, cur_pixel[0]] >= hand_brightnes:
            cur_pixel[1] = cur_pixel[1] - step*2
        elif frame_out[cur_pixel[1] - step*2, cur_pixel[0] + step*2] >= hand_brightnes:
            cur_pixel[1] = cur_pixel[1] - step*2
            cur_pixel[0] = cur_pixel[0] + step*2
        elif frame_out[cur_pixel[1] - step*2, cur_pixel[0] - step*2] >= hand_brightnes:
            cur_pixel[1] = cur_pixel[1] - step*2
            cur_pixel[0] = cur_pixel[0] - step*2
        else:
            cur_pixel[1] = 79
            cur_pixel[0] = 80
            hand_present_flag = False

while True:
    check, frame = video.read()     # Create a frame object
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # Converting to grayscale

    for i in range(0, height_len, factor):
        for j in range(0, width_len, factor):
            frame_out[int((i/factor))][int(j/factor)] = gray[i][j]

    cv2.rectangle(frame_out, (0, 0), (60, 15), 0, -1)

    if hand_present_flag:
        find_lowest(bright_pixel)
        cv2.rectangle(frame_out, (lowest_pixel[0]-20, lowest_pixel[1]-40), (lowest_pixel[0]+20, lowest_pixel[1]), 255, 1)

        frame_extract = cv2.getRectSubPix(frame_out, (38, 38), (lowest_pixel[0], lowest_pixel[1]-19))  # Extracting frame
        frame_extract.resize([1, 38, 38, 1])
        prediction_result_h = model_h.predict(frame_extract)  # Prediction for hand/no hand

        if prediction_result_h[0][0] < prediction_result_h[0][1] and prediction_result_h[0][1] >= 0.9:
            prediction_result_g = model_g.predict(frame_extract)  # Prediction for gestures
            if prediction_result_g[0][0] >= 0.7:
                prediction_text = "Fist"
            elif prediction_result_g[0][1] >= 0.7:
                prediction_text = "Palm"
            elif prediction_result_g[0][2] >= 0.7:
                prediction_text = "Finger"
            else:
                prediction_text = "Sth went wrong"
        else:
            prediction_text = "No hand"

        cv2.putText(frame_out, prediction_text, (0, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.9, color=255, thickness=1)
    else:
        prediction_text = "Activate"
        cv2.putText(frame_out, prediction_text, (0, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.9, color=255, thickness=1)
        for pixel in pixels_to_check:                   # Searching for bright pixel
            if frame_out[pixel[1], pixel[0]] >= hand_brightnes:
                bright_pixel = pixel
                lowest_pixel = pixel
                hand_present_flag = True
                find_lowest(bright_pixel)
                break

        cv2.rectangle(frame_out, (lowest_pixel[0] - 20, lowest_pixel[1] - 40), (lowest_pixel[0] + 20, lowest_pixel[1]), 255, 1)

    cv2.imshow("Hello World!", frame_out)  # Show the frame
    key = cv2.waitKey(1)  # For playing

    if key == ord('q'):
        break

# Shutdown the camera
video.release()
cv2.destroyAllWindows()
