from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d
import scipy.io
import os
import tensorflow as tf
import numpy as np
import random
import math


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. Do not modify the constructor, as doing so 
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()
        #(num_examples, 32, 32, 3)

        self.batch_size = 64
        self.num_classes = 2
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []
        # TODO: Initialize all hyperparameters
        self.learning_rate = 0.001

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        conv1 = tf.nn.conv2d(
            inputs, filters=self.filter1, strides=[1, 2, 2, 1], padding='SAME') + self.filter1_bias

        batch_norm1 = tf.nn.batch_normalization(conv1,
                                                mean=tf.nn.moments(
                                                    conv1, axes=[0, 1, 2])[0],
                                                variance=tf.nn.moments(
                                                    conv1, axes=[0, 1, 2])[1],
                                                offset=None, scale=None, variance_epsilon=1e-10)

        relu1 = tf.nn.relu(batch_norm1)
        max_pool1 = tf.nn.max_pool(relu1, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
        conv2 = tf.nn.conv2d(
            max_pool1, self.filter2, [1, 1, 1, 1], 'SAME') + self.filter2_bias

        batch_norm2 = tf.nn.batch_normalization(conv2,
                                                mean=tf.nn.moments(
                                                    conv2, axes=[0, 1, 2])[0],
                                                variance=tf.nn.moments(
                                                    conv2, axes=[0, 1, 2])[1],
                                                offset=None, scale=None, variance_epsilon=1e-10)
        relu2 = tf.nn.relu(batch_norm2)
        max_pool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 1, 1, 1], 'SAME')
        if is_testing:
            conv3 = conv2d(max_pool2, self.filter3, [
                           1, 1, 1, 1], 'SAME') + self.filter3_bias
        else:
            conv3 = tf.nn.conv2d(
                max_pool2, self.filter3, [1, 1, 1, 1], 'SAME') + self.filter3_bias

        batch_norm3 = tf.nn.batch_normalization(conv3,
                                                mean=tf.nn.moments(
                                                    conv3, axes=[0, 1, 2])[0],
                                                variance=tf.nn.moments(
                                                    conv3, axes=[0, 1, 2])[1],
                                                offset=None, scale=None, variance_epsilon=1e-10)
        relu3 = tf.nn.relu(batch_norm3)

        relu3 = tf.reshape(relu3, [relu3.shape[0], -1])
        dl1 = tf.matmul(relu3, self.W1) + self.b1
        dl1 = tf.nn.dropout(dl1, 0.3)
        dl2 = tf.matmul(dl1, self.W2) + self.b2
        dl2 = tf.nn.dropout(dl2, 0.3)
        dl3 = tf.matmul(dl2, self.W3) + self.b3
        return dl3

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''
    losses = []

    indices = [i for i in range(train_inputs.shape[0])]
    indices = tf.random.shuffle(indices)
    train_inputs = tf.gather(train_inputs, indices)
    train_labels = tf.gather(train_labels, indices)

    for i in range(0, train_inputs.shape[0]//model.batch_size*model.batch_size, model.batch_size):
        image = train_inputs[i: i + model.batch_size, :, :, :]
        label = train_labels[i:i + model.batch_size]
        # Implement backprop:
        with tf.GradientTape() as tape:
            predictions = model(image)
            loss = model.loss(predictions, label)
            losses.append(loss)
            if i//model.batch_size % 4 == 0:
                train_acc = model.accuracy(model(image), label)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

    return losses


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    # Loop through 5000 training images
    Acc = 0
    n = 0
    for i in range(0, test_inputs.shape[0], model.batch_size):
        image = test_inputs[i:i + model.batch_size, :, :, :]
        label = test_labels[i:i + model.batch_size]

        predictions = model(image)  # this calls the call function conveniently
        loss = model.loss(predictions, label)
        train_acc = model.accuracy(model(
            test_inputs[i:i + model.batch_size, :, :, :]), test_labels[i:i + model.batch_size])
        n += 1
        Acc += train_acc

    print('Test accuracy: {}'.format(Acc/n))
    return Acc/n


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            "{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 

    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.

    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.

    :return: None
    '''
    input_file = './data/train'
    test_input_file = './data/test'

    inputs, labels = get_data(input_file, 3, 5)
    test_inputs, test_labels = get_data(
        test_input_file, 3, 5)
    model = Model()
    # TODO: Train model by calling train() ONCE on all data
    num_epochs = 25
    for _ in range(num_epochs):  # MOVE THIS
        train(model, inputs, labels)
    # TODO: Test the accuracy by calling test() after running train()
    acc = test(model, test_inputs, test_labels)
    print(acc)


if __name__ == '__main__':
    main()
