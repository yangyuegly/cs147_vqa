import preprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
import spacy

class VQA(tf.keras.Model):
    def __init__(self):
        # self.annotations = annotations
        # self.questions = questions
        # self.answers = None

        # Hyperparameters
        self.batch_size = 128
        self.hidden_size = 1024
        self.rnn_size = 512
        self.output_size = 1000
        self.embedding_size = 300
        self.nlp = spacy.load("en_core_web_md")
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Trainable parameters
        self.img_ff_layer = Dense(self.hidden_size)
        self.lstm_hidden = LSTM(self.rnn_size, return_sequence=True, return_state=True)
        self.lstm = LSTM(self.rnn_size, return_state=True)
        self.txt_ff_layer = Dense(self.hidden_size, activation='tanh')

        # Fuse/Merge Layers
        self.merge_layer1 = Dense(self.output_size, activation='relu')
        self.merge_layer2 = Dense(self.output_size)
    
    def call(self, img_feats, ques_inputs):
        """
        Runs a forward pass on image and question tensors
        :param img_inputs: tensor of image features with shape [batch_size x 4096]
        :param ques_inputs: a list of strings with length [batch_size]
        :return: probabilities
        """
        # Image part
        # L2-normalizing the image tensor TODO: check if axis=1
        normalized_img_feats = tf.math.l2_normalize(img_feats, axis=1, epsilon=1e-12, name=None)
        img_output = self.img_ff_layer(normalized_img_feats)
        
        # LSTM part
        window_size = max([len(q.split()) for q in ques_inputs])
        embeddings = np.zeros((self.batch_size, window_size, self.embedding_size))
        for i in range(ques_inputs):
            doc = self.nlp(ques_inputs[i])
            for t in len(doc):
                if t < window_size:
                    embeddings[i, t, :] = np.reshape(doc[t].vector, [1, self.embedding_size])
        hidden_output, hidden_memory_state, hidden_carry_state = self.lstm_hidden(embeddings, initial_state=None)
        _, final_memory_state, final_carry_state = self.lstm(hidden_output, initial_state=(hidden_memory_state, hidden_carry_state))
        txt_output = self.txt_ff_layer(tf.concat([final_memory_state, final_carry_state], axis=1))

        # Fusing
        fused = tf.multiply(img_output, txt_output)
        fuse_output1 = self.merge_layer1(fused)
        logits = self.merge_layer2(fuse_output1)
        probabilities = tf.nn.softmax(logits)

        return probabilities
        
    def loss(self, probabilities, labels):
        """
        Computes the cross-entropy loss given y_true (labels) and y_pred (probabilities)
        :return: the average loss across a batch of inputs
        """
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)
        return tf.reduce_mean(losses)

def train(model, img_feats, ques_inputs, labels):
    """
    :param img_feats: 2-D tensor of shape [number_images x 4096]
    :param ques_inputs: string of length [number_images]
    :param labels: 
    """
    indices = np.arange(img_feats.shape[0])
    indices = tf.random.shuffle(indices)
    
    bsz = model.batch_size
    input_size = img_feats.shape[0]
    for i in range(0,input_size, bsz):
        img_batch = img_feats[i:(i+bsz),:]
        ques_batch = ques_inputs[i:(i+bsz)]
        labels_batch = labels[i:(i+bsz)]

        with tf.GradientTape() as tape:
            probs = model.call(img_batch, ques_batch)
            batch_loss = model.loss(probs, labels_batch)
        
        gradients = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, img_inputs, ques_inputs, labels):
    """
    :param img_feats: 2-D tensor of shape [number_images x 4096]
    :param ques_inputs: string of length [number_images]
    :param labels: 
    :returns: loss of the epoch
    """
    probs = model.call(img_inputs, ques_inputs)
    loss = model.loss(probs, labels)
    return loss

def main():
    fpath_anno = "./data/mscoco_train2014_annotations.json"
    fpath_q_mc = "./data/MultipleChoice_mscoco_train2014_questions.json"
    annotations_mc, questions_mc = preprocess.load_text(fpath_anno, fpath_q_mc)
    questions_text_mc = [x['question'] for x in questions_mc] # ques_inputs for train   
    vqa_mc = VQA()
    