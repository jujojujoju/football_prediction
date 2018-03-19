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


def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def transform_result(data):
    data['FTR'] = data.apply(lambda x: 1
    if (x['FTR'] == 'H')
    else(2 if (x['FTR'] == 'D') else 0), axis=1)
    return pd.DataFrame(data)


min_max_scalar = MinMaxScaler()
match_data = pd.DataFrame()
# data from laliga 07~17
# for i in range(2, 10):
#     match_data = match_data.append(pd.read_csv('./laliga_recent/SP1 (' + str(i + 1) + ').csv'))

path = '/Users/joju/PycharmProjectst/tensorflowTutorial/laliga_recent/'
extension = 'csv'
os.chdir(path)
result = [i for i in glob.glob('*.{}'.format(extension))]
print(result)
test_match_data = pd.DataFrame()
for file in result:
    if (file == 'SP1 (1).csv'):
        test_match_data = pd.read_csv(path + file)

    match_data = match_data.append(pd.read_csv(path + file))

rows = ["BbMxH", "BbAvH", "BbMxD", "BbAvD", "BbMxA", "BbAvA",
        "B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA",
        "FTR"]

total_row_num = match_data.shape[0]
# train_row_num = int(match_data.shape[0] * 0.8)
train_row_num = int(match_data.shape[0] * 1)

# # subset중 하나라도 missing value 이면 제거한다
match_data.dropna(subset=rows, inplace=True)
match_data = match_data[rows]
print(match_data.shape[0])

train_match_data = match_data.head(train_row_num)
# test_match_data = match_data.tail(total_row_num - train_row_num)

rows = ["BbMxH", "BbAvH", "BbMxD", "BbAvD", "BbMxA", "BbAvA",
        "B365H", "B365D", "B365A", "BWH", "BWD", "BWA", "IWH", "IWD", "IWA",
        "FTR"]

data = list(set(test_match_data['HomeTeam']))
team_name = pd.DataFrame({'Point': [0]}, index=data)
# print(team_name)

test_match_data.dropna(subset=rows, inplace=True)

test_match_data_with_team = test_match_data
test_match_data_with_team = test_match_data_with_team.values

print(test_match_data_with_team)
print(test_match_data_with_team[0, 2])
print(test_match_data_with_team[0, 3])
print(team_name.ix[test_match_data_with_team[0, 2], 0])

# print(team_name)
print(test_match_data_with_team.shape[0])

test_match_data = test_match_data[rows]

print(train_match_data.shape, test_match_data.shape)

train_match_data = transform_result(train_match_data).values
test_match_data = transform_result(test_match_data).values

print(train_match_data)

x_data = train_match_data[:, 0:-1]
x_data = min_max_scalar.fit_transform(x_data)
y_data = train_match_data[:, [-1]]

test_x_data = test_match_data[:, 0:-1]
test_x_data = min_max_scalar.fit_transform(test_x_data)
test_y_data = test_match_data[:, [-1]]

attr = x_data.shape[1]
output = y_data.shape[1]

print(x_data.shape, y_data.shape)

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

W6 = tf.get_variable("W6", shape=[attr, nb_classes],
                     initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(L5, W6) + b6
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

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(train_row_num / batch_size)

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
print('Accuracy:', sess.run(accuracy, feed_dict={
    X: test_x_data, Y: test_y_data, keep_prob: 1}))

predicted_team_name = pd.DataFrame()
predicted_team_name = team_name
predicted_team_name = 0
# Get one and predict
for r in range(test_match_data_with_team.shape[0]):
    # r = random.randint(0, total_row_num - train_row_num - 1)
    label = sess.run(tf.argmax(test_y_data[r:r + 1], 1))
    prediction = sess.run(tf.argmax(hypothesis, 1), feed_dict={X: test_x_data[r:r + 1], keep_prob: 1})
    if(label == 1):
        team_name.ix[test_match_data_with_team[r,2], 0] += 3
    elif label == 2:
        team_name.ix[test_match_data_with_team[r,2], 0] += 1
        team_name.ix[test_match_data_with_team[r,3], 0] += 1
    else:
        team_name.ix[test_match_data_with_team[r,3], 0] += 3

    if (prediction == 1):
        predicted_team_name.ix[test_match_data_with_team[r, 2], 0] += 3
    elif prediction == 2:
        predicted_team_name.ix[test_match_data_with_team[r, 2], 0] += 1
        predicted_team_name.ix[test_match_data_with_team[r, 3], 0] += 1
    else:
        predicted_team_name.ix[test_match_data_with_team[r, 3], 0] += 3

    print(r)
    print(test_match_data_with_team[r,2], test_match_data_with_team[r,3])
    print("Label: ", label)
    print("Prediction: ", prediction)

team_name.sort_values(by=['Point'], axis=0, inplace=True, ascending=False)
predicted_team_name.sort_values(by=['Point'], axis=0, inplace=True, ascending=False)
predicted_team_name['Rank'] = range(1, predicted_team_name.shape[0] + 1, 1)
team_name['Rank'] = range(1, predicted_team_name.shape[0] + 1, 1)
print("team_name : \n", team_name)
print("predicted_team_name : \n", predicted_team_name)
