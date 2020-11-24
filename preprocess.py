import json
import cv2
import numpy as np
import os
from PIL import Image


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

def load_img(dir_path):
    images = []
    for fname in os.listdir(dir_path):
        img = Image.open(os.path.join(dir_path, fname))
        images.append(np.asarray(img))
    
    return np.array(images)
        


if __name__ == "__main__":
    fpath_anno = "./data/mscoco_train2014_annotations.json"
    fpath_q_mc = "./data/MultipleChoice_mscoco_train2014_questions.json"
    load_text(fpath_anno, fpath_q_mc)

