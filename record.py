import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')       # Define a codec
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

ret, frame = cap.read()
height_len = len(frame)  # Height of input image
width_len = len(frame[0])  # Width of input image

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        cv2.rectangle(frame, (int(width_len/2)-int(height_len/6), int(height_len/3)), (int(width_len/2)+int(height_len/6), int(height_len/3)*2), (0,0,255), 3)
        cv2.imshow('Recording', frame)

        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()