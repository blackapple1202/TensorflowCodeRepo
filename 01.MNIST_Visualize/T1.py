
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import random as ran

# 시각화를 위한 변수 및 함수들

SHOW_TRAIN_EXAMPLE = 1
SHOW_PREDICTION = 20

def display_trainData(num, train_size):
    x_train = mnist.train.images[:train_size, :]
    y_train = mnist.train.labels[:train_size, :]
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Train Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


def display_prediction(num):
    # num번째 test 데이터를 불러온다
    x_test = mnist.test.images[num,:].reshape(1,784)
    y_test = mnist.test.labels[num,:]
    # 정답을 label 에 넣어놓는다.
    label = y_test.argmax()
    # 예측값을 우리 모델을 이용해서 연산한다.
    # 연산 후 argmax를 통해 예측한 label을 prediction에 넣는다.
    # prediction이랑 label을 비교 해본다.
    prediction = sess.run(hypothesis, feed_dict={X: x_test}).argmax()
    plt.title('Test Example: %d Prediction: %d Label: %d' % (num, prediction, label))
    plt.imshow(x_test.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

# 1. Construction 단계

# MNIST 이미지 데이터
X = tf.placeholder(tf.float32, [None, 784])
# MNIST 라벨 데이터
Y = tf.placeholder(tf.float32, [None, 10])

# 가중치(Weight)
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Softmax를 이용한 hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

# 정확도 검증하기 위한 Operation
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 2. Execution 단계

for _ in range(SHOW_TRAIN_EXAMPLE):
    display_trainData(ran.randint(0, 55000), 55000)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
print("Learning Start")
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={X: batch_xs, Y: batch_ys})
        print('Step %d, Training accuracy %g' % (i, train_accuracy))

    sess.run(train_step, feed_dict={X: batch_xs, Y: batch_ys})
print("Learning finished")


print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))


for _ in range(SHOW_PREDICTION):
    display_prediction(ran.randint(0, 10000))

sess.close()

