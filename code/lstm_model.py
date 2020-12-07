import numpy as np
import tensorflow as tf
import spacy
from tensorflow.keras.layers import LSTM, Dense

class LSTMmodel(tf.keras.Model):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.rnn_size = 512
        self.num_output = 1024
        self.nlp = spacy.load("en_core_web_md")
        self.embedding_size = 300
        self.lstm_hidden = LSTM(self.rnn_size, return_sequence=True, return_state=True)
        self.lstm = LSTM(self.rnn_size, return_state=True)
        self.text_dense = Dense(self.num_output, activation='tanh')
    
    def call(self, batched_questions):
        '''
        :param batched_questions: a list of strings
        :return: a 2d tensor with shape [batch_size, num_output]
        '''
        window_size = max([len(q.split()) for q in batched_questions])
        embeddings = np.zeros((self.batch_size, window_size, self.embedding_size))
        for i in range(batched_questions):
            doc = self.nlp(batched_questions[i])
            for t in len(doc):
                if t < window_size:
                    embeddings[i, t, :] = np.reshape(doc[t].vector, [1, self.embedding_size])
        hidden_output, hidden_memory_state, hidden_carry_state = self.lstm_hidden(embeddings, initial_state=None)
        _, final_memory_state, final_carry_state = self.lstm(hidden_output, initial_state=(hidden_memory_state, hidden_carry_state))
        return self.dense(tf.concat([final_memory_state, final_carry_state], axis=1))
