from preprocess import preprocess
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
import spacy
from keras.models import model_from_yaml
import argparse


class VQA(tf.keras.Model):
    def __init__(self, vocab_size):
        # self.annotations = annotations
        # self.questions = questions
        # self.answers = None

        # Hyperparameters
        self.batch_size = 128
        self.hidden_size = 1024
        self.rnn_size = 512
        self.merge_hidden_size = 1000
        self.embedding_size = 300
        self.vocab_size = vocab_size
        self.nlp = spacy.load("en_core_web_md")
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        # Trainable parameters
        self.img_ff_layer = Dense(self.hidden_size)
        self.lstm_hidden = LSTM(self.rnn_size, return_sequence=True, return_state=True)
        self.lstm = LSTM(self.rnn_size, return_state=True)
        self.txt_ff_layer = Dense(self.hidden_size, activation='tanh')

        # Fuse/Merge Layers
        self.merge_layer1 = Dense(self.merge_hidden_size, activation='relu')
        self.merge_layer2 = Dense(self.vocab_size)
    
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
    :param labels: a list of answer_id in the vocab
    :returns: a list of losses of all batches
    """
    # Shuffling
    indices =  [i for i in range(len(ques_inputs))]
    indices = tf.random.shuffle(indices)
    img_feats = tf.gather(img_feats, indices)
    shuffled_questions = []
    shuffled_labels = []
    for index in indices.numpy():
        shuffled_questions.append(ques_inputs[index])
        shuffled_labels.append(labels[index])

    # Train batches
    bsz = model.batch_size
    input_size = img_feats.shape[0]
    losses = []
    for i in range(0,input_size, bsz):
        batch_imgs = img_feats[i:(i+bsz),:]
        batch_questions = shuffled_questions[i:(i+bsz)]
        batch_labels = shuffled_labels[i:(i+bsz)]

        with tf.GradientTape() as tape:
            probs = model.call(batch_imgs, batch_questions)
            batch_loss = model.loss(probs, batch_labels)
        
        gradients = tape.gradient(batch_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        losses.append(batch_loss)
    
    return losses

def test(model, img_inputs, ques_inputs, labels):
    """
    :param img_feats: 2-D tensor of shape [number_images x 4096]
    :param ques_inputs: string of length [number_images]
    :param labels:
    :returns: loss of the epoch
    """
    probs = model(img_inputs, ques_inputs)
    loss = model.loss(probs, labels)
    return loss

def loss_visualization(losses):
    """
    :param losses: A list of losses across batches
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def run(img_flag):
    fpath_train_anno = "../data/annotations/mscoco_train2014_annotations.json"
    fpath_train_q_mc = "../data/questions/MultipleChoice_mscoco_train2014_questions.json"
    fpath_train_img_dir = "../data/train2014/"

    with open(fpath_train_anno, 'r') as f:
        train_annotations_raw = json.load(f)
    
    train_annotations = train_annotations_raw['annotations']
    vocab = build_answer_vocab(train_annotations)

    fpath_test_anno = "../data/annotations/mscoco_test2014_annotations.json"
    fpath_test_q_mc = "../data/questions/MultipleChoice_mscoco_test2014_questions.json"
    fpath_test_img_dir = "../data/test2014/"

    # annotations_mc, questions_mc = preprocess.load_text(fpath_anno, fpath_q_mc)
    questions_train, labels_train, img_features_train = preprocess(fpath_train_anno,fpath_train_q_mc,img_flag,vocab,dir_path_image=fpath_train_img_dir,category=0) 
    questions_test, labels_test, img_features_test = preprocess(fpath_test_anno,fpath_test_q_mc,img_flag,fpath_test_img_dir,vocab,category=2)    
    
    vocab_size = len(vocab)
    vqa_mc = VQA(vocab_size)
    losses = train(vqa_mc, img_features, questions, labels)
    test(model, img_inputs, ques_inputs, labels)

    # Save Model
    model.save('./model')
    # loaded_model = tf.keras.models.load_model('/tmp/model')
    print("Saved model to disk")
    
    # later...
    
    test =


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQA Model")
    parser.add_argument("-i","--img", help="")
    args = parser.parse_args()

    if args.i:
        print("run and store image features")
        run(True)
    else:
        print("run without preprocessing image")
        run(False)
