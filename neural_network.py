import tensorflow as tf
from tensorflow import keras
import numpy as np
from glob import glob
import os
import cv2
import re


def atoi(text):
    return int(text) if text.isdigit() else text

# alist.sort(key=natural_keys) sorts in human order
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def nn_go():
    print("Preparing arrays...")
    train_images_folder_path = "C:\\Users\\wojte\\Desktop\\Praca_inz\\train_images"
    train_images_paths = glob(os.path.join(train_images_folder_path, "*.jpg"))
    train_images = np.array([[[0] * 38] * 38] * 1937, dtype=np.uint8)
    train_images_labels = np.array([0] * 1937)

    train_noise_folder_path = "C:\\Users\\wojte\\Desktop\\Praca_inz\\train_noise"
    train_noise_paths = glob(os.path.join(train_noise_folder_path, "*.jpg"))
    train_noise = np.array([[[0]*38]*38]*1937, dtype=np.uint8)

    train_hand = np.array([[[0]*38]*38]*3874, dtype=np.uint8)
    train_hand_labels = np.array([0]*3874)

    img_path = "C:\\Users\\wojte\\Desktop\\Praca_inz\\test_images"
    img_paths = glob(os.path.join(img_path, "*.jpg"))
    img_test = np.array([[[0] * 38] * 38] * 9, dtype=np.uint8)

    # Sort paths in their arrays
    print("Sorting paths...")
    train_images_paths.sort(key=natural_keys)
    train_noise_paths.sort(key=natural_keys)
    img_paths.sort(key=natural_keys)

    # Converting train images from jpg to numpy
    print("Converting images to numpy...")
    for i in range(len(train_images_paths)):
        train_images[i] = cv2.cvtColor(cv2.imread(train_images_paths[i]), cv2.COLOR_BGR2GRAY)
        train_hand[i] = train_images[i]

    # Converting noise from jpg to numpy
    for i in range(len(train_noise_paths)):
        train_noise[i] = cv2.cvtColor(cv2.imread(train_noise_paths[i]), cv2.COLOR_BGR2GRAY)
        train_hand[1937+i] = train_noise[i]

    # Converting test images from jpg to numpy
    for i in range(len(img_paths)):
        img_test[i] = cv2.cvtColor(cv2.imread(img_paths[i]), cv2.COLOR_BGR2GRAY)

    # Creating labels for train images: numpy array with values 0-2
    print("Creating labels...")

    for i in range(len(train_images_labels)):
        if i < 739:
            train_images_labels[i] = 0
        elif i < 1345:
            train_images_labels[i] = 1
        else:
            train_images_labels[i] = 2


    # Creating labels for train hand: numpy array with values 0-1
    for i in range(len(train_hand_labels)):
        if i < 1937:
            train_hand_labels[i] = 0
        else:
            train_hand_labels[i] = 1

    print("Scaling data...")
    train_images = train_images / 255.0
    train_hand = train_hand / 255.0

    #############################################################
    # Hand recognition training
    print("Preparing trainings...")

    layer_conv2d_2_1 = keras.layers.Conv2D(8, (8, 8), input_shape=(38, 38, 1), activation=tf.nn.relu)
    layer_conv2d_2_2 = keras.layers.Conv2D(16, (8, 8), activation=tf.nn.relu)
    layer_maxpooling2d_2_1 = keras.layers.MaxPooling2D(pool_size=(8, 8))
    layer_maxpooling2d_2_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
    layer_maxpooling2d_2_3 = keras.layers.MaxPooling2D(pool_size=(1, 1))
    layer_flatten_2_1 = keras.layers.Flatten()
    layer_dense_2_1 = keras.layers.Dense(50, activation=tf.nn.relu)
    layer_dense_2_2 = keras.layers.Dense(20, activation=tf.nn.relu)
    layer_dense_2_3 = keras.layers.Dense(2, activation=tf.nn.softmax)

    model_h = keras.Sequential([
        layer_conv2d_2_1,
        layer_conv2d_2_2,
        layer_maxpooling2d_2_1,
        layer_maxpooling2d_2_2,
        layer_maxpooling2d_2_3,
        layer_flatten_2_1,
        layer_dense_2_1,
        layer_dense_2_2,
        layer_dense_2_3
    ])

    model_h.compile(optimizer=keras.optimizers.Adam(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    train_hand.resize([3874, 38, 38, 1])

    ##############################################################
    # Gesture recognition training

    layer_conv2d_1 = keras.layers.Conv2D(8, (4, 4), input_shape=(38, 38, 1), activation=tf.nn.relu)
    layer_conv2d_2 = keras.layers.Conv2D(4, (4, 4), activation=tf.nn.relu)
    layer_maxpooling2d = keras.layers.MaxPooling2D(pool_size=(2, 2))
    layer_flatten = keras.layers.Flatten()
    layer_dense_1 = keras.layers.Dense(50, activation=tf.nn.relu)
    layer_dense_2 = keras.layers.Dense(50, activation=tf.nn.relu)
    layer_dense_3 = keras.layers.Dense(3, activation=tf.nn.softmax)

    model_g = keras.Sequential([
        layer_conv2d_1,
        layer_conv2d_2,
        layer_maxpooling2d,
        layer_flatten,
        layer_dense_1,
        layer_dense_2,
        layer_dense_3
    ])

    model_g.compile(optimizer=keras.optimizers.Adam(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    train_images.resize([1937, 38, 38, 1])
    img_test.resize([9, 38, 38, 1])

    ########################### T R A I N I N G #######################################

    model_h.fit(train_hand, train_hand_labels, epochs=3)
    model_g.fit(train_images, train_images_labels, epochs=1)

    # Saving entire model to a HDF5 file
    # model_h.save('F:\\PyCharm 5.0.4\\PROJEKTY\\ExternalCamera\\model_h.h5')
    # model_g.save('F:\\PyCharm 5.0.4\\PROJEKTY\\ExternalCamera\\model_g.h5')

    ############################# Summary ####################################

    model_h.summary()
    model_g.summary()

    ########################## Tests ########################

    img_test = img_test / 255.0
    prediction_frame_h = model_h.predict(img_test)
    print(prediction_frame_h)
    prediction_frame_g = model_g.predict(img_test)
    print(prediction_frame_g)

    print("Returning models.")
    return model_h, model_g
