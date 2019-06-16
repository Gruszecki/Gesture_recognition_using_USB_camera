from glob import glob
import os
import cv2
from matplotlib import pyplot as plt

img_counter = 0
img_paths = None
img = None
img_prev = None
img_next = None
fig, (axprev, axcur, axnext) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 3, 1]})

def convert_rgb_mode(image):   # changing BGR to RGB
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    return image

def open_img():
    global img_counter, img

    axprev.axis('off')
    axprev.imshow(convert_rgb_mode(img_prev))
    axprev.set_title("Prev")
    axcur.axis('off')
    axcur.imshow(convert_rgb_mode(img))
    axnext.axis('off')
    axnext.imshow(convert_rgb_mode(img_next))
    axnext.set_title("Next")

    plt.ion()  # interactive mode
    plt.draw()
    plt.pause(0.001)

def init(dir_path):
    global img, img_prev, img_next, img_paths
    img_paths = glob(os.path.join(dir_path, "*.jpg"))
    img = cv2.imread(img_paths[img_counter])  # load path
    img_prev = cv2.imread(img_paths[len(img_paths) - 1])
    img_next = cv2.imread(img_paths[img_counter + 1])
    open_img()

def next_image(hand_shift):
    global img, img_prev, img_next, img_counter, img_paths
    if hand_shift == -1:
        if img_counter == 0:
            img_counter = len(img_paths) - 1
            img_prev = cv2.imread(img_paths[img_counter - 1])
            img_next = cv2.imread(img_paths[0])
        else:
            img_counter = img_counter - 1
            img_next = cv2.imread(img_paths[img_counter + 1])
            if img_counter == 0:
                img_prev = cv2.imread(img_paths[len(img_paths) - 1])
            else:
                img_prev = cv2.imread(img_paths[img_counter - 1])
    elif hand_shift == 1:
        if len(img_paths) - 1 != img_counter:
            img_counter = img_counter + 1
            img_prev = cv2.imread(img_paths[img_counter - 1])
            if img_counter == len(img_paths) - 1:
                img_next = cv2.imread(img_paths[0])
            else:
                img_next = cv2.imread(img_paths[img_counter + 1])
        else:
            img_counter = 0
            img_prev = cv2.imread(img_paths[len(img_paths) - 1])
            img_next = cv2.imread(img_paths[img_counter + 1])
    img = cv2.imread(img_paths[img_counter])  # load path
    open_img()
