
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
import matplotlib.pyplot as plt
import random as ran

# model 경로
save_file = './model/T1_cnn/T1_cnnModel.ckpt'

# 시각화를 위한 변수 및 함수들

SHOW_TRAIN_EXAMPLE = 1
SHOW_PREDICTION = 10

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
    prediction = sess.run(hypothesis, feed_dict={X: x_test, keep_prob: 0.5}).argmax()
    plt.title('Test Example: %d Prediction: %d Label: %d' % (num, prediction, label))
    plt.imshow(x_test.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

# Weight 초기화 함수
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

# Bias 초기화 함수
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# conv2d 함수
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# max_pool 함수
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 1. Construction 단계

# MNIST 이미지 데이터
X = tf.placeholder(tf.float32, [None, 784])
# MNIST 라벨 데이터
Y = tf.placeholder(tf.float32, [None, 10])

# 1번째 Convolution 층 생성
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(X, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2번째 Convolution 층 생성
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# 층 연결
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout 계산
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout 층 생성
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# hypothesis 계산
hypothesis = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=hypothesis))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Model을 save 하기 위한 변수
# 보통 학습시키는 요소(통상 Variable)들만 저장한다. 이 예제의 경우 W, b이다.
saver = tf.train.Saver()


# 2. Execution 단계

for _ in range(SHOW_TRAIN_EXAMPLE):
    display_trainData(ran.randint(0, 55000), 55000)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 이미 학습되었던 CNN 기반 모델
print("Model Load Begin")
saver.restore(sess, save_file)
print("Model Load End")

print('Test accuracy %g' % accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0}))


for _ in range(SHOW_PREDICTION):
    display_prediction(ran.randint(0, 10000))

sess.close()
