import tensorflow as tf
from tensorflow import keras
import numpy as np
from glob import glob
import os
import cv2
import re


def nn_go():
    print("Preparing arrays...")
    train_images_folder_path = "C:\\Users\\wojte\\Desktop\\Praca_inz\\train_images"
    train_images_paths = glob(os.path.join(train_images_folder_path, "*.png"))
    train_images = np.array([[[0] * 38] * 38] * 1937, dtype=np.uint8)
    train_images_labels = np.array([0] * 1937)
    train_noise_folder_path = "C:\\Users\\wojte\\Desktop\\Praca_inz\\train_noise"
    train_noise_paths = glob(os.path.join(train_noise_folder_path, "*.jpg"))
    train_noise = np.array([[[0]*38]*38]*6000, dtype=np.uint8)
    train_hand = np.array([[[0]*38]*38]*7937, dtype=np.uint8)
    train_hand_labels = np.array([0]*7937)
    # test_images_folder_path = "C:\\Users\\wojte\\Desktop\\Praca_inz\\test_images"
    # test_images_paths = glob(os.path.join(test_images_folder_path, "*.jpg"))
    # test_images = np.array([[[0]*38]*38]*727, dtype=np.uint8)
    # test_images_labels = np.array([0]*727)
    # test_noise_folder_path = "C:\\Users\\wojte\\Desktop\\Praca_inz\\test_noise"
    # test_noise_paths = glob(os.path.join(test_noise_folder_path, "*.jpg"))
    # test_noise = np.array([[[0]*38]*38]*727, dtype=np.uint8)
    # test_noise_labels = np.array([0]*80)
    # test_hand = np.array([[[0]*38]*38]*807, dtype=np.uint8)
    # test_hand_labels = np.array([0]*807)
    # class_names = ['Fist', 'Palm', 'Gun', 'Two', 'Thumb_Left', 'Satan', 'Point_Thumb', 'Little', 'Hand_Left', 'Three']

    img_path = "C:\\Users\\wojte\\Desktop\\Praca_inz\\test_images"
    img_paths = glob(os.path.join(img_path, "*.jpg"))
    img_test = np.array([[[0] * 38] * 38] * 9, dtype=np.uint8)

    def atoi(text):
        return int(text) if text.isdigit() else text

    # alist.sort(key=natural_keys) sorts in human order
    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    # Sort paths in their arrays
    print("Sorting paths...")
    train_images_paths.sort(key=natural_keys)
    train_noise_paths.sort(key=natural_keys)
    # test_images_paths.sort(key=natural_keys)
    # test_noise_paths.sort(key=natural_keys)
    img_paths.sort(key=natural_keys)

    print("Converting images to numpy...")
    # Converting train images from jpg to numpy
    for i in range(len(train_images_paths)):
        train_images[i] = cv2.cvtColor(cv2.imread(train_images_paths[i]), cv2.COLOR_BGR2GRAY)
        train_hand[i] = train_images[i]

    # Converting noise from jpg to numpy
    for i in range(len(train_noise_paths)):
        train_noise[i] = cv2.cvtColor(cv2.imread(train_noise_paths[i]), cv2.COLOR_BGR2GRAY)
        train_hand[1937+i] = train_noise[i]

    # # Converting test images from jpg to numpy
    # for i in range(len(test_images_paths)):
    #     test_images[i] = cv2.cvtColor(cv2.imread(test_images_paths[i]), cv2.COLOR_BGR2GRAY)
    #     test_hand[i] = test_images[i]
    #
    # # Converting test noise from jpg to numpy
    # for i in range(len(test_noise_paths)):
    #     test_noise[i] = cv2.cvtColor(cv2.imread(test_noise_paths[i]), cv2.COLOR_BGR2GRAY)
    #     test_hand[727+i] = test_noise[i]

    for i in range(len(img_paths)):
        img_test[i] = cv2.cvtColor(cv2.imread(img_paths[i]), cv2.COLOR_BGR2GRAY)

    print("Creating labels...")
    # Creating labels for train images: numpy array with values 0-2
    label_value = 0
    for i in range(len(train_images_labels)):
        if i == 739 or i == 1345:
            label_value = label_value + 1
        train_images_labels[i] = label_value

    # Creating labels for train hand: numpy array with values 0-1
    label_value = 1
    for i in range(len(train_hand_labels)):
        if i == 1937:
            label_value = 0
        train_hand_labels[i] = label_value

    # # Creating labels for test images: numpy array with values 0-9
    # label_value = 0
    # for i in range(len(test_images_labels)):
    #     if i == 79 or i == 129 or i == 203 or i == 288 or i == 349 or i == 440 or i == 518 or i == 595 or i == 673:
    #         label_value = label_value + 1
    #     test_images_labels[i] = label_value
    #
    # # Creating labels for test hand: numpy array with values 0-1
    # label_value = 1
    # for i in range(len(test_hand_labels)):
    #     if i == 727:
    #         label_value = 0
    #     test_hand_labels[i] = label_value

    print("Scaling data...")
    train_images = train_images / 255.0
    # train_noise = train_noise / 255.0
    train_hand = train_hand / 255.0
    # test_images = test_images / 255.0
    # test_noise = test_noise / 255.0
    # test_hand = test_hand / 255.0

    print("Training...")
    #############################################################
    # Hand recognition training

    layer_conv2d_2_1 = keras.layers.Conv2D(8, (4, 4), input_shape=(38, 38, 1), activation=tf.nn.relu)
    layer_conv2d_2_2 = keras.layers.Conv2D(16, (4, 4), activation=tf.nn.relu)
    layer_maxpooling2d_2_1 = keras.layers.MaxPooling2D(pool_size=(1, 1))
    layer_flatten_2_1 = keras.layers.Flatten()
    layer_dense_2_1 = keras.layers.Dense(50, activation=tf.nn.relu)
    layer_dense_2_2 = keras.layers.Dense(50, activation=tf.nn.relu)
    layer_dense_2_3 = keras.layers.Dense(2, activation=tf.nn.softmax)

    model_h = keras.Sequential([
        layer_conv2d_2_1,
        layer_conv2d_2_2,
        layer_maxpooling2d_2_1,
        layer_flatten_2_1,
        layer_dense_2_1,
        layer_dense_2_2,
        layer_dense_2_3
    ])

    model_h.compile(optimizer=tf.train.AdamOptimizer(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    train_hand.resize([7937, 38, 38, 1])
    model_h.fit(train_hand, train_hand_labels, epochs=1)

    ##############################################################
    # Gesture recognition training

    layer_conv2d_1 = keras.layers.Conv2D(32, (4, 4), input_shape=(38, 38, 1), activation=tf.nn.relu)
    layer_conv2d_2 = keras.layers.Conv2D(64, (4, 4), activation=tf.nn.relu)
    layer_maxpooling2d = keras.layers.MaxPooling2D(pool_size=(1, 1))
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

    model_g.compile(optimizer=tf.train.AdamOptimizer(),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    train_images.resize([1937, 38, 38, 1])
    img_test.resize([9, 38, 38, 1])
    model_g.fit(train_images, train_images_labels, epochs=1)

    #############################################################
    # Summary

    model_h.summary()
    model_g.summary()

    # # Saving weights
    # weights_file = open("C:\\Users\\wojte\\Desktop\\Praca_inz\\weights.txt", 'w')
    # weights_file.write(str(layer_conv2d_1.get_weights()))
    # weights_file.write(str(layer_conv2d_2.get_weights()))
    # weights_file.write(str(layer_maxpooling2d.get_weights()))
    # weights_file.write(str(layer_flatten.get_weights()))
    # weights_file.write(str(layer_dense_1.get_weights()))
    # weights_file.write(str(layer_dense_2.get_weights()))
    # weights_file.write(str(layer_dense_3.get_weights()))
    # weights_file.close()

    ##############################################################
    # Tests

    img_test = img_test / 255.0
    prediction_frame_h = model_h.predict(img_test)
    print(prediction_frame_h)
    prediction_frame_g = model_g.predict(img_test)
    print(prediction_frame_g)

    print("Returning models.")
    return model_h, model_g
