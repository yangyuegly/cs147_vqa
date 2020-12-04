import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras import Model
BATCH_SIZE = 128


def load_text(fpath_anno, fpath_q_mc, img_filenames):
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
    #obtain list of labels
    for question in annotations:
        image_id_list.append(question["image_id"])
        question_id_list.append(question["question_id"])
        labels = labels.append(question["multiple_choice_answer"])
    
    #build question dictionary from json file
    for question in questions_mc:
        question_dict[question["question_id"]] = question["question"]
    
    #find corresponding question to each question id 
    for question in question_id_list:
        questions.append(question_dict[question])
        
        
    print("Questions MC: ", len(questions_mc))
    print("Annotations: ", len(annotations))


    return questions, labels, image_id_list


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

    base_model = VGG19(include_top=True, weights='imagenet')
    model = Model(input=base_model.input,
                  output=base_model.get_layer('fc2').output)
    img_filenames = []
    for img_id in image_id_list:
        num_of_zeros = 12 - len(img_id)  #curr id length 
        zeros = num_of_zeros * "0" 
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

    return img_features, img_filenames 


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

# def extract_question_features(model, questions):
#     num_questions = len(questions)
#     batch_feat_list = []

#     # Batching
#     for i in range(0, num_questions, BATCH_SIZE):
#         question_batch = questions[i:(i+BATCH_SIZE)]
#         batch_features = model.call(question_batch)
#         batch_feat_list.append(batch_features)
    
#     question_features = np.concatenate(np.array(batch_feat_list), axis=0)
#     return question_features

if __name__ == "__main__":
    fpath_anno = "./data/mscoco_train2014_annotations.json"
    fpath_q_mc = "./data/MultipleChoice_mscoco_train2014_questions.json"
    annotations, questions_mc = load_text(fpath_anno, fpath_q_mc)
    questions_text_mc = [x['question'] for x in questions_mc]