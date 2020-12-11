# cs147_vqa
VQA project for CS147

This repository aims to tackle visual question answering tasks. Current function is limited to multiple choice questions.

How to run our model:
  1. Run with image preprocessing:
     If this is your first time running our model, then you may not have images extracted as features. In this case, you may want to train the model while extracting image features using the VGG-19 network pre-trained on ImageNet.
     Command: python ./vqa.py -i
  2. Run without image preprocessing:
     If this is not your first time running our model and you have image features already extracted in a folder called "weights_features", then you can simply train the rest of the model using the command below:
     python ./vqa.py
