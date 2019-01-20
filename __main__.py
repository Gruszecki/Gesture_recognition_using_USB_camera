import cv2
import numpy as np
nn = __import__('neural_network')

model_h, model_g = nn.nn_go()

# Create an object.
video = cv2.VideoCapture(0)

check, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

height_len = len(gray)      # Height of input image
width_len = len(gray[0])    # Width of input image
factor = 4                  # Factor used to change dimension
avg_color = 0               # Value used for image with changed dimension
frames_dims = [[80, 60], [60, 60], [100, 60], [80, 40], [80, 80], [60, 40], [100, 40], [60, 80], [100, 80]]

frame_out = np.array([[0]*(int(width_len/factor))]*int((height_len/factor)), dtype=np.uint8)    # Table with new dimension

while True:
    check, frame = video.read()     # Create a frame object
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # Converting to grayscale

    for i in range(0, height_len, factor):
        for j in range(0, width_len, factor):
            frame_out[int((i/factor))][int(j/factor)] = gray[i][j]

    for dim in frames_dims:
        frame_extract = cv2.getRectSubPix(frame_out, (38, 38), (dim[0], dim[1]))  # Extracting frame
        frame_extract.resize([1, 38, 38, 1])
        prediction_result_h = model_h.predict(frame_extract)  # Prediction for hand/no hand

        if prediction_result_h[0][0] < prediction_result_h[0][1] and prediction_result_h[0][1] >= 0.975:
            cv2.rectangle(frame_out, (dim[0]-20, dim[1]-20), (dim[0]+20, dim[1]+20), 255, 1)

            prediction_result_g = model_g.predict(frame_extract)  # Prediction for gestures
            if prediction_result_g[0][0] >= 0.7:
                prediction_text = "Fist"
            elif prediction_result_g[0][1] >= 0.7:
                prediction_text = "Palm"
            elif prediction_result_g[0][2] >= 0.7:
                prediction_text = "Finger"
            else:
                prediction_text = "Sth went wrong"

            break
        else:
            prediction_text = "No hand"

    cv2.rectangle(frame_out, (0, 0), (60, 15), 0, -1)
    cv2.putText(frame_out, prediction_text, (0, 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.9, color=255, thickness=1)
    cv2.imshow("Hello World!", frame_out)  # Show the frame
    key = cv2.waitKey(1)  # For playing

    if key == ord('q'):
        break

# Shutdown the camera
video.release()
cv2.destroyAllWindows()
