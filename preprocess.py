import json
import cv2
import numpy as np
import os
from tf.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Flatten
import tensorflow as tf
from tf.keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
BATCH_SIZE = 128


def load_text(fpath_anno, fpath_q_mc):
    with open(fpath_anno, 'r') as f1:
        annotations_raw = json.load(f1)
    with open(fpath_q_mc, 'r') as f2:
        questions_mc_raw = json.load(f2)

    annotations = annotations_raw['annotations']
    questions_mc = questions_mc_raw['questions']

    print("Questions MC: ", len(questions_mc))
    print("Annotations: ", len(annotations))

    return annotations, questions_mc


def preprocess_img(dir_path):
    """
    Given a dir_path containing all images, loads and preprocesses all images
    and then extracts features using a pre-trained VGG19 model
    :param: path to a directory containing all images
    :return: an array of image features of shape (num_images, 4096)
    """
    images = []

    base_model = VGG19(include_top=True, weights='imagenet')
    model = Model(input=base_model.input,
                  output=base_model.get_layer('fc2').output)

    for fname in os.listdir(dir_path):
        img = load_img(os.path.join(dir_path, fname),
                       color='rgb', target_size=[224, 224])
        img = img_to_array(img)
        images.append(img)

    images = np.vstack(images)  # Shape: (num_images, 224, 224, 3)

    num_images = images.shape[0]
    batch_feat_list = []

    # Batching
    for i in range(0, num_images, BATCH_SIZE):
        image_batch = images[i: i + BATCH_SIZE, :, :, :]
        batch_features = extract_image_features(model, image_batch)
        batch_feat_list.append(batch_features)

    img_features = np.concatenate(np.array(batch_feat_list), axis=0)
    return img_features


def extract_image_features(model, images):
    """
    Uses the pre-trained VGG19 network to extract features from an array of images
    :param img:
    :return: 4096-dimensional feature vectors
    """
    inputs = preprocess_input(images)
    features = model.predict(inputs)  # Shape: (num_images, 4096)
    return features


def save_image(features):
    image_features = open('weights_features/image_features.txt')
    image_features.write(features)


if __name__ == "__main__":
    fpath_anno = "./data/mscoco_train2014_annotations.json"
    fpath_q_mc = "./data/MultipleChoice_mscoco_train2014_questions.json"
    load_text(fpath_anno, fpath_q_mc)
