import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras import Model
BATCH_SIZE = 128


def build_answer_vocab(annotations):
    """
    Builds a vocab
    :param: annotations for the training set
    """
    all_words = []
    for question in annotations:
        all_words.append(question["multiple_choice_answer"])
        
    all_words = sorted(set(all_words))
    vocab = {word: i for i, word in enumerate(all_words)}
    return vocab

def load_text(fpath_anno, fpath_q_mc, vocab):
    with open(fpath_anno, 'r') as f1:
        annotations_raw = json.load(f1)
    with open(fpath_q_mc, 'r') as f2:
        questions_mc_raw = json.load(f2)

    annotations = annotations_raw['annotations']
    questions_mc = questions_mc_raw['questions']
    question_id_list = []
    image_id_list = []
    labels = []
    questions = []
    questions_dict = {}

    # Obtain list of labels
    for question in annotations:
        image_id_list.append(question["image_id"])
        question_id_list.append(question["question_id"])
        labels.append(vocab[question["multiple_choice_answer"]])
    
    # Build question dictionary from json file
    for question in questions_mc:
        questions_dict[question["question_id"]] = question["question"]
    
    # Find corresponding question to each question id 
    for question in question_id_list:
        questions.append(questions_dict[question])

    vocab_size = len(vocab)
    return questions, labels, image_id_list, vocab_size


def preprocess_img(dir_path, image_id_list, category=None):
    """
    Given a dir_path containing all images, loads and preprocesses all images
    and then extracts features using a pre-trained VGG19 model
    :param: path to a directory containing all images
    :param: image_id_list a list of 
    :param: category 0 == train , 1 == validation, 2 == test; None if using 2017 datasets

    :return: an array of image features of shape (num_images, 4096)
    """
    images = []

    if category == 0:
        prefix = "COCO_train2014_"
    elif category == 1:
        prefix = "COCO_val2014_"
    elif category == 2:
        prefix = "COCO_test2014_"
    else:
        prefix = ""

    # Initialize VGG model for feature extraction 
    base_model = VGG19(include_top=True, weights='imagenet')
    model = Model(input=base_model.input,
                  output=base_model.get_layer('fc2').output)

    # Load corresponding images according to img_id_list and convert to img array 
    for img_id in image_id_list:
        num_zeros = 12 - len(img_id)  # Current id length 
        zeros = num_zeros * "0" 
        curr_filename = prefix + zeros + img_id
        img = load_img(os.path.join(dir_path, curr_filename),
                       color_mode='rgb', target_size=[224, 224])
                    
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


def save_image(features, category):
    filename = open('weights_features/image_features_' + category + '.txt')
    np.savetxt(filename, features, fmt='%d')


def preprocess(fpath_anno, fpath_q_mc, img_flag, vocab, dir_path_image=None, category=None):
    print("Preprocessing...")
    questions, labels, image_id_list, vocab_size = load_text(fpath_anno, fpath_q_mc, vocab)
    if img_flag:
        img_features = preprocess_img(dir_path_image, image_id_list, category=category)
        save_image(img_features, category)
        print("Image features saved.")
    else:
        img_features = np.loadtxt('weights_features/image_features_' + category + '.txt', dtype=np.int32)
        print("Image features loaded.")

    print("Preprocessing complete (๑ `▽´๑)")
    return questions, labels, vocab_size, img_features