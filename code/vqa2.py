from preprocess import preprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
import spacy
from tensorflow.keras.models import model_from_yaml
import argparse
import json
from matplotlib import pyplot as plt


class VQA(tf.keras.Model):
    def __init__(self):
        super(VQA, self).__init__()
        # Hyperparameters
        self.batch_size = 32
        self.hidden_size = 1024
        self.rnn_size = 512
        self.merge_hidden_size = 1000
        self.embedding_size = 300
        self.nlp = spacy.load("en_core_web_md")
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Trainable parameters
        self.img_ff_layer = Dense(
            self.hidden_size, kernel_initializer='uniform')
        self.lstm_hidden = LSTM(
            self.rnn_size, return_sequences=True, return_state=True)
        self.lstm = LSTM(self.rnn_size, return_state=True)
        self.txt_ff_layer = Dense(self.hidden_size, activation='tanh')

        # Fuse/Merge Layers
        self.merge_layer1 = Dense(
            self.merge_hidden_size, kernel_initializer='uniform', activation='relu')
        self.merge_layer2 = Dense(self.embedding_size)

    def call(self, img_feats, ques_inputs):
        """
        Runs a forward pass on image and question tensors
        :param img_inputs: tensor of image features with shape [batch_size, 4096]
        :param ques_inputs: a list of strings with shape [batch_size,]
        :return: logits with shape [batch_size, embedding_size]
        """
        # Image part
        print("image feats: ", img_feats.shape)
        img_output = self.img_ff_layer(img_feats)

        # LSTM part
        window_size = max([len(q.split()) for q in ques_inputs])
        embeddings = np.random.normal(0, 0.1, size=(self.batch_size, window_size, self.embedding_size))
        for i in range(len(ques_inputs)):
            doc = self.nlp(ques_inputs[i])
            for t in range(len(doc)):
                if t < window_size:
                    embeddings[i, t, :] = np.reshape(
                        doc[t].vector, [1, self.embedding_size])
        embeddings = tf.cast(embeddings, dtype=tf.float32)
        hidden_output, hidden_memory_state, hidden_carry_state = self.lstm_hidden(
            embeddings, initial_state=None)
        _, final_memory_state, final_carry_state = self.lstm(
            hidden_output, initial_state=(hidden_memory_state, hidden_carry_state))
        txt_output = self.txt_ff_layer(
            tf.concat([final_memory_state, final_carry_state], 1))

        # Fusing
        fused = tf.multiply(img_output, txt_output)
        fuse_output1 = self.merge_layer1(fused)
        logits = self.merge_layer2(fuse_output1)

        return logits

    def loss_function(self, logits, labels):
        y_true = [self.nlp(answer).vector for answer in labels]
        # y_true = tf.math.l2_normalize(y_true)
        # logits = tf.math.l2_normalize(logits)
        cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
        return tf.reduce_mean(cosine_loss(tf.convert_to_tensor(y_true), logits))

    def predict(self, logits):
        return None


def train(model, img_feats, ques_inputs, labels):
    """
    :param img_feats: 2-D tensor of shape [number_images x 4096]
    :param ques_inputs: string of length [number_images]
    :param labels: a list of string representing answers
    :returns: a list of losses of all batches
    """
    # Shuffling
    indices = [i for i in range(len(img_feats))]  # TEST
    indices = tf.random.shuffle(indices)
    img_feats = tf.gather(img_feats, indices, axis=0)
    shuffled_questions = []
    shuffled_labels = []
    
    for index in indices.numpy():
        shuffled_questions.append(ques_inputs[index])
        shuffled_labels.append(labels[index])

    # Train batches
    bsz = model.batch_size
    input_size = img_feats.shape[0]
    losses = []
    num_iteration = input_size // bsz
    for i in range(num_iteration):
        batch_imgs = img_feats[i*bsz:(i+1)*bsz, :]
        batch_questions = shuffled_questions[i*bsz:(i+1)*bsz]
        batch_labels = shuffled_labels[i*bsz:(i+1)*bsz]

        with tf.GradientTape() as tape:
            logits = model.call(batch_imgs, batch_questions)
            batch_loss = model.loss_function(logits, batch_labels)
        print("Iteration loss: ", batch_loss.numpy())

        gradients = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))
        losses.append(batch_loss.numpy())

        # Save Model
        model.save_weights('./model_weights.h5',overwrite=True)
    
    return losses


def validate(model, img_inputs, ques_inputs, labels):
    """
    :param img_feats: 2-D tensor of shape [number_images x 4096]
    :param ques_inputs: string of length [number_images]
    :param labels:
    :returns: loss of the epoch
    """
    probs = model(img_inputs, ques_inputs)
    loss = model.loss_function(probs, labels)
    return loss


def visualize_loss(losses):
    """
    :param losses: A list of losses across batches
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def visualize_results():
    predict()
    print()


def run(img_flag):
    fpath_train_anno = "../data/annotations/mscoco_train2014_annotations.json"
    fpath_train_q_mc = "../data/questions/MultipleChoice_mscoco_train2014_questions.json"
    fpath_train_img_dir = "../data/train2014/"

    with open(fpath_train_anno, 'r') as f:
        train_annotations_raw = json.load(f)

    train_annotations = train_annotations_raw['annotations']

    fpath_val_anno = "../data/annotations/mscoco_val2014_annotations.json"
    fpath_val_q_mc = "../data/questions/MultipleChoice_mscoco_val2014_questions.json"
    fpath_val_img_dir = "../data/val2014/"

    questions_train, labels_train, img_features_train, _ = preprocess(
        fpath_train_anno, fpath_train_q_mc, img_flag, dir_path_image=fpath_train_img_dir, category=0)
    # questions_val, labels_val, img_features_val = preprocess(
    #     fpath_val_anno, fpath_val_q_mc, img_flag, fpath_val_img_dir, vocab, category=2)

    vqa_mc = VQA()
    losses = train(vqa_mc, img_features_train, questions_train, labels_train)
    
    visualize_loss(losses)

    # Save Model
    vqa_mc.save_weights('./model_weights.h5',overwrite=True)
    # loaded_model = tf.keras.models.load_model('/tmp/model')
    print("Saved model weights to disk")

    # validate(vqa_mc, img_features_val, questions_val, labels_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQA Model")
    parser.add_argument("-i", "--img", action='store_true',
                        help="", default="")
    args = vars(parser.parse_args())

    if args["img"]:
        print("Run and store image features")
        run(True)
    else:
        print("Run without preprocessing image")
        run(False)
