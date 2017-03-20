
# coding: utf-8

import gzip
import os

import numpy as np
import scipy.spatial.distance as sp_dist
import random
import math
import tensorflow as tf

from sklearn.cross_validation import KFold

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances

import scipy.special as ss

vectorTxtFile = "Q1/glove.6B.300d.txt"
Q4List = "Q4/wordList.csv"

fast_text_path = "Q4/fastText_vectors.txt"
lazaridou_path = "Q4/vector_lazaridou.txt"

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])     # Hidden layer with RELU activation
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])       # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']      # Output layer with linear activation
    return out_layer

#function to create mini_batches while training the MLP model
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]

    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)

    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]

#function to perform k_fold_validation (here k=5)
def k_fold_valid_function(X,Y):
    scores = []
    # print(len(X), len(Y))
    kf = KFold(n = len(X), n_folds=5)
    for train_index, test_index in kf:
        # print(train_index, test_index)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        yield X_train, X_test, Y_train, Y_test

#function to compute cosine sim between vec1 and vec2
def cosine_sim(vec1, vec2) :
    return cosine_similarity([vec1],[vec2])[0,0]

def derivedWordTask(inputFile = Q4List):
    fast_file = open(fast_text_path, 'r')
    laz_file = open(lazaridou_path, 'r')

    fast_vec_dict = {}
    fast_vec_len = 0

    laz_vec_dict = {}
    laz_vec_len = 0

    for file_row in fast_file:
        curr_line = file_row.strip().split()
        curr_word = curr_line[0]
        curr_vec = []
        for index, elem in enumerate(curr_line):
            if index == 0 :
                continue
            curr_vec.append(float(elem.strip()))
        fast_vec_len = len(curr_vec)
        fast_vec_dict[curr_word] = curr_vec

    # print(len(fast_vec_dict), fast_vec_len)

    for file_row in laz_file:
        curr_line = file_row.strip().split()
        curr_word = curr_line[0]
        curr_vec = []
        for index, elem in enumerate(curr_line):
            if index == 0:
                continue
            curr_vec.append(float(elem.strip("[], ")))
        laz_vec_len = len(curr_vec)
        laz_vec_dict[curr_word] = curr_vec

    # print(len(laz_vec_dict), laz_vec_len)

    deriv_pairs = []
    affix_set = set()

    wordList_file = open(Q4List, 'r').read().strip().split("\n")[1:]

    for file_row in wordList_file:
        curr_line = file_row.strip().split(',')
        index_val = int(curr_line[0])
        curr_affix = curr_line[1]
        curr_derived = curr_line[2]
        curr_base = curr_line[3]
        affix_set.add(curr_affix)
        deriv_pairs.append((curr_base, curr_derived, curr_affix))

    # print deriv_pairs

    affix_int_dict = {}
    curr_int = 0

    for item in affix_set:
        affix_int_dict[item] = curr_int
        curr_int = curr_int + 1

    num_affix = len(affix_set)
    # print(num_affix)

    fast_ft_vec = np.empty((len(deriv_pairs), num_affix+fast_vec_len))
    fast_out = np.empty((len(deriv_pairs), fast_vec_len))

    for index, item in enumerate(deriv_pairs):
        if (item[0] not in fast_vec_dict) or (item[1] not in fast_vec_dict) :
            continue
        ft_vec1 = fast_vec_dict[item[0]]
        curr_affix_val = item[2]
        ft_vec2 = [0]*num_affix
        ft_vec2[affix_int_dict[curr_affix_val]] = 1
        fast_ft_vec[index,:] = np.hstack((ft_vec1, ft_vec2))
        fast_out[index,:] = np.asarray(fast_vec_dict[item[1]])

    # print(np.shape(fast_ft_vec), np.shape(fast_out))

    learning_rate = 0.001
    training_epochs = 20
    batch_size = 100
    display_step = 1

    n_hidden_1 = 300 # 1st layer number of features
    n_hidden_2 = 300 # 2nd layer number of features
    n_input = fast_vec_len+num_affix
    n_classes = fast_vec_len

    k_fold_valid_num = 0

    out_prediction = np.empty((0, fast_vec_len))

    for curr_f_num in k_fold_valid_function(fast_ft_vec, fast_out):
        X_train, X_test, y_train, y_test = curr_f_num
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])*np.sqrt(2./(n_input+n_hidden_1))),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])*np.sqrt(2./(n_hidden_1+n_hidden_2))),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])*np.sqrt(2./(n_hidden_2+n_classes)))
        }
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'out': tf.Variable(tf.zeros([n_classes]))
        }
        # Construct model
        pred = multilayer_perceptron(x, weights, biases)
        # Define loss and optimizer
        cost = tf.reduce_sum((pred-y)*(pred-y))/(2*batch_size)
        # cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                # total_batch = int((len(pos_train)+len(neg_train))/batch_size)
                # Loop over all batches
                for batch in iterate_minibatches(X_train, y_train, batch_size):
                # for i in range(total_batch):
                    batch_x, batch_y = batch
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    # Compute average loss
                    avg_cost += c / batch_size
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            k_fold_valid_num = k_fold_valid_num + 1
            print("Fold Iteration : ",k_fold_valid_num," : Optimization Finished")

            # Testing the model
            test_data_x = tf.placeholder("float", [None, n_input])
            test_data_pred = multilayer_perceptron(test_data_x, weights, biases)

            test_data_pred = sess.run(test_data_pred, feed_dict = {test_data_x:X_test})
            out_prediction = np.vstack((out_prediction, test_data_pred))

    # print(np.shape(out_prediction))

    cosVal1 = 0

    fast_file_fp = open( "Q4/AnsFastText.txt", "w")
    ans_fast_file_fp = open( "Q4/AnsModel_FastText.txt", "w")

    for i  in range(np.shape(out_prediction)[0]):
        cosVal1 += cosine_sim(out_prediction[i,:], fast_out[i,:])
        str_fast = str(deriv_pairs[i][1])
        str_fast_ans = str(deriv_pairs[i][1])
        for j in range(np.shape(out_prediction)[1]):
            str_fast = str_fast + str(" ") + str(out_prediction[i][j])
            str_fast_ans = str_fast_ans + str(" ") + str(fast_out[i][j])
        str_fast = str_fast + "\n"
        str_fast_ans = str_fast_ans + "\n"
        fast_file_fp.write(str_fast)
        ans_fast_file_fp.write(str_fast_ans)

    fast_file_fp.close()
    ans_fast_file_fp.close()

    cosVal1 = cosVal1/np.shape(out_prediction)[0]
    print("FastText CosSim value : ", cosVal1)

    #LAZ case
    laz_ft_vec = np.empty((len(deriv_pairs), num_affix+laz_vec_len))
    laz_out = np.empty((len(deriv_pairs), laz_vec_len))

    for index, item in enumerate(deriv_pairs):
        if (item[0] not in laz_vec_dict) or (item[1] not in laz_vec_dict) :
            continue
        ft_vec1 = laz_vec_dict[item[0]]
        curr_affix_val = item[2]
        ft_vec2 = [0]*num_affix
        ft_vec2[affix_int_dict[curr_affix_val]] = 1

        laz_ft_vec[index,:] = np.hstack((ft_vec1, ft_vec2))
        laz_out[index,:] = np.asarray(laz_vec_dict[item[1]])

    # print(np.shape(laz_ft_vec), np.shape(laz_out))

    learning_rate = 0.001
    training_epochs = 20
    batch_size = 100
    display_step = 1

    n_hidden_1 = 300 # 1st layer number of features
    n_hidden_2 = 300 # 2nd layer number of features
    n_input = laz_vec_len+num_affix
    n_classes = laz_vec_len

    k_fold_valid_num = 0

    laz_out_prediction = np.empty((0, laz_vec_len))

    for curr_f_num in k_fold_valid_function(laz_ft_vec, laz_out):
        X_train, X_test, y_train, y_test = curr_f_num
        # tf Graph input
        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, n_classes])
        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])*np.sqrt(2./(n_input+n_hidden_1))),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])*np.sqrt(2./(n_hidden_1+n_hidden_2))),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])*np.sqrt(2./(n_hidden_2+n_classes)))
        }
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden_1])),
            'b2': tf.Variable(tf.zeros([n_hidden_2])),
            'out': tf.Variable(tf.zeros([n_classes]))
        }
        # Construct model
        pred = multilayer_perceptron(x, weights, biases)
        # Define loss and optimizer
        cost = tf.reduce_sum((pred-y)*(pred-y))/(2*batch_size)
        # cost = tf.reduce_mean(tf.losses.mean_squared_error(predictions=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                # total_batch = int((len(pos_train)+len(neg_train))/batch_size)
                # Loop over all batches
                for batch in iterate_minibatches(X_train, y_train, batch_size):
                # for i in range(total_batch):
                    batch_x, batch_y = batch
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    # Compute average loss
                    avg_cost += c / batch_size
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

            k_fold_valid_num = k_fold_valid_num + 1
            print("Fold Iteration : ",k_fold_valid_num," : Optimization Finished")

            # Testing the model
            test_data_x = tf.placeholder("float", [None, n_input])
            test_data_pred = multilayer_perceptron(test_data_x, weights, biases)

            test_data_pred = sess.run(test_data_pred, feed_dict = {test_data_x:X_test})
            laz_out_prediction = np.vstack((laz_out_prediction, test_data_pred))

    # print(np.shape(laz_out_prediction))

    cosVal2 = 0

    laz_file_fp = open( "Q4/AnsLzaridou.txt", "w")
    ans_laz_file_fp = open( "Q4/AnsModel_Lzaridou.txt", "w")

    for i  in range(np.shape(laz_out_prediction)[0]):
        cosVal2 += cosine_sim(laz_out_prediction[i,:], laz_out[i,:])
        str_laz = str(deriv_pairs[i][1])
        str_laz_ans = str(deriv_pairs[i][1])
        for j in range(np.shape(laz_out_prediction)[1]):
            str_laz = str_laz + str(" ") + str(laz_out_prediction[i][j])
            str_laz_ans = str_laz_ans + str(" ") + str(laz_out[i][j])
        str_laz = str_laz + "\n"
        str_laz_ans = str_laz_ans + "\n"
        laz_file_fp.write(str_laz)
        ans_laz_file_fp.write(str_laz_ans)

    laz_file_fp.close()
    ans_laz_file_fp.close()

    cosVal2 = cosVal2/np.shape(laz_out_prediction)[0]
    print("Lazaridou CosSim value : ", cosVal2)

    return cosVal1, cosVal2

def main():
    derCos1,derCos2 = derivedWordTask()

if __name__ == '__main__':
    main()
