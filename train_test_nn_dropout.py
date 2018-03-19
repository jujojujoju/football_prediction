import sqlite3
import csv
import pandas as pd

pd.set_option('display.width', 320)
import os
import numpy as np
import glob
import tensorflow as tf
import random

tf.set_random_seed(777)  # for reproducibility
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import warnings

warnings.simplefilter("ignore")

nb_classes = 3  # 0 ~ 2

# parameters
learning_rate = 0.001
training_epochs = 1
batch_size = 500
avg_rate = 2


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


min_max_scalar = MinMaxScaler()
match_data = pd.DataFrame()

# xy = np.loadtxt('match_train_odd.csv', delimiter=';', dtype=np.float32)
# test_xy = np.loadtxt('match_test_odd.csv', delimiter=';', dtype=np.float32)
xy = np.loadtxt('match_train_odd_player.csv', delimiter=';', dtype=np.float32)
test_xy = np.loadtxt('match_test_odd_player.csv', delimiter=';', dtype=np.float32)

x_data = xy[:, 0:-1]
x_data = min_max_scalar.fit_transform(x_data)
y_data = xy[:, [-1]]

test_x_data = test_xy[:, 0:-1]
test_x_data = min_max_scalar.fit_transform(test_x_data)
test_y_data = test_xy[:, [-1]]

attr = x_data.shape[1]
output = y_data.shape[1]

# input place holders
X = tf.placeholder(tf.float32, [None, attr])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

hidden_attr = 500

# weights & bias for nn layers
# http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
W1 = tf.get_variable("W1", shape=[attr, hidden_attr],
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([hidden_attr]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[hidden_attr, hidden_attr],
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([hidden_attr]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[hidden_attr, hidden_attr],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([hidden_attr]))
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.get_variable("W4", shape=[hidden_attr, attr],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([attr]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

W5 = tf.get_variable("W5", shape=[attr, attr],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([attr]))
L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)

W6 = tf.get_variable("W6", shape=[attr, attr],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([attr]))
L6 = tf.nn.relu(tf.matmul(L5, W6) + b6)
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)

W7 = tf.get_variable("W7", shape=[attr, attr],
                     initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([attr]))
L7 = tf.nn.relu(tf.matmul(L6, W7) + b7)
L7 = tf.nn.dropout(L7, keep_prob=keep_prob)

W8 = tf.get_variable("W8", shape=[attr, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L7, W8) + b8

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

onehot_encoder = OneHotEncoder(sparse=False)
y_data = onehot_encoder.fit_transform(y_data.reshape(len(y_data), 1))
test_y_data = onehot_encoder.fit_transform(test_y_data.reshape(len(test_y_data), 1))


data = list(['Win', 'Draw', 'Defeat'])
result = pd.DataFrame(columns=data, index=data).fillna(0)
result_Array = {'label': '', 'prediction': ''}
avg_result = pd.DataFrame(columns=data, index=data).fillna(0)
avg_final_accuracy = 0

for k in range(avg_rate):
    # train my model
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(x_data.shape[0] / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size, x_data, y_data)
            feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Get one and predict
    for r in range(test_xy.shape[0]):
        # r = random.randint(0, total_row_num - train_row_num - 1)
        label = sess.run(tf.argmax(test_y_data[r:r + 1], 1))
        prediction = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: test_x_data[r:r + 1], keep_prob: 1})
        result_Array = {'label':'','prediction':''}
        if (label == 1):
            result_Array['label'] = 'Win'
        elif label == 2:
            result_Array['label'] = 'Draw'
        else:
            result_Array['label'] = 'Defeat'

        if (prediction == 1):
            result_Array['prediction'] = 'Win'
        elif prediction == 2:
            result_Array['prediction'] = 'Draw'
        else:
            result_Array['prediction'] = 'Defeat'

        result.ix[result_Array['label'],result_Array['prediction']] += 1

        print(str(r + 1) + "st match label: ", label)
        print(str(r + 1) + "st match Prediction: ", prediction)

    result /= (test_xy.shape[0])
    print(result)
    avg_result += result
    final_accuracy = sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data, keep_prob: 1})
    print('Accuracy:', final_accuracy)
    avg_final_accuracy += final_accuracy

win_precision = avg_result.ix['Win', 'Win'] / (
    avg_result.ix['Win', 'Win'] + avg_result.ix['Draw', 'Win'] + avg_result.ix['Defeat', 'Win'])
win_recall = avg_result.ix['Win', 'Win'] / (
    avg_result.ix['Win', 'Win'] + avg_result.ix['Win', 'Draw'] + avg_result.ix['Win', 'Defeat'])
draw_precision = avg_result.ix['Draw', 'Draw'] / (
    avg_result.ix['Win', 'Draw'] + avg_result.ix['Draw', 'Draw'] + avg_result.ix['Defeat', 'Draw'])
draw_recall = avg_result.ix['Draw', 'Draw'] / (
    avg_result.ix['Draw', 'Win'] + avg_result.ix['Draw', 'Draw'] + avg_result.ix['Draw', 'Defeat'])
defeat_precision = avg_result.ix['Defeat', 'Defeat'] / (
    avg_result.ix['Win', 'Defeat'] + avg_result.ix['Draw', 'Defeat'] + avg_result.ix['Defeat', 'Defeat'])
defeat_recall = avg_result.ix['Defeat', 'Defeat'] / (
    avg_result.ix['Defeat', 'Win'] + avg_result.ix['Defeat', 'Draw'] + avg_result.ix['Defeat', 'Defeat'])
columns = list(['Win', 'Draw', 'Defeat'])
indexs = list(['precision', 'recall'])
recall_precision = pd.DataFrame(columns=columns, index=indexs).fillna(0)
recall_precision.ix['precision', 'Win'] = win_precision
recall_precision.ix['precision', 'Draw'] = draw_precision
recall_precision.ix['precision', 'Defeat'] = defeat_precision
recall_precision.ix['recall', 'Win'] = win_recall
recall_precision.ix['recall', 'Draw'] = draw_recall
recall_precision.ix['recall', 'Defeat'] = defeat_recall
recall_precision = recall_precision.fillna(0)
avg_result /= avg_rate
avg_final_accuracy /= avg_rate
print("avg_result : \n", avg_result)
print("recall_precision : \n", recall_precision)
print("avg_final_accuracy : \n", avg_final_accuracy)

