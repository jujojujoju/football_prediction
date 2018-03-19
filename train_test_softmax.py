import tensorflow as tf
import numpy as np
import pandas as pd

tf.set_random_seed(777)  # for reproducibility
from sklearn.preprocessing import MinMaxScaler

avg_rate = 5
learning_rate = 0.01
steps = 30000

# Predicting animal type based on various features
xy = np.loadtxt('match_train_odd.csv', delimiter=';', dtype=np.float32)
# test_xy = np.loadtxt('match_test_odd.csv', delimiter=';', dtype=np.float32)
test_xy = np.loadtxt('realTest_odd.csv', delimiter=';', dtype=np.float32)

x_data = xy[:, 0:-1]
print(x_data)
min_max_scalar = MinMaxScaler()
x_data = min_max_scalar.fit_transform(x_data)
y_data = xy[:, [-1]]

test_x_data = test_xy[:, 0:-1]
test_x_data = min_max_scalar.fit_transform(test_x_data)
test_y_data = test_xy[:, [-1]]

attr = x_data.shape[1]
output = y_data.shape[1]

print(x_data.shape, y_data.shape)

nb_classes = 3  # 0 ~ 2

X = tf.placeholder(tf.float32, [None, attr])
Y = tf.placeholder(tf.int32, [None, output])  # 0 ~ 6
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([attr, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)
# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

data = list(['Win', 'Draw', 'Defeat'])
result = pd.DataFrame(columns=data, index=data).fillna(0)
result_Array = {'label': '', 'prediction': ''}
avg_result = pd.DataFrame(columns=data, index=data).fillna(0)
avg_final_accuracy = 0

# Launch graph
with tf.Session() as sess:
    for k in range(avg_rate):
        sess.run(tf.global_variables_initializer())

        for step in range(steps):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            if step % 100 == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
                print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

        pred = sess.run(prediction, feed_dict={X: test_x_data})
        ok = 0
        count = 0
        for p, y in zip(pred, test_y_data.flatten()):
            count += 1
            print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
            if (p == int(y)):
                ok += 1
            else:
                print(count, test_x_data[count - 1, :], int(y))

            if (int(y) == 1):
                result_Array['label'] = 'Win'
            elif int(y) == 2:
                result_Array['label'] = 'Draw'
            else:
                result_Array['label'] = 'Defeat'

            if (p == 1):
                result_Array['prediction'] = 'Win'
            elif p == 2:
                result_Array['prediction'] = 'Draw'
            else:
                result_Array['prediction'] = 'Defeat'

            result.ix[result_Array['label'], result_Array['prediction']] += 1

        result /= (test_xy.shape[0])
        print(result)
        avg_result += result
        final_accuracy = (ok / count) * 100
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
    print(avg_result)
    print(recall_precision)
    print(avg_final_accuracy)
    print("avg_result : \n", avg_result)
    print("recall_precision : \n", recall_precision)
    print("avg_final_accuracy : \n", avg_final_accuracy)
