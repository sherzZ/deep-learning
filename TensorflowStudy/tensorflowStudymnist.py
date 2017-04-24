#coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# ����mnist ����
# one_hot���� one-hot �����ǽ���Ӧ��ʵ����ĵ�Ԫ��Ϊ��Ϊ 1������Ԫ��Ϊ 0
mnist = input_data.read_data_sets('../../datasets/MNIST_data', one_hot = True)

print("Training data size ",mnist.train.num_examples)

print("Validating data size ",mnist.validation.num_examples)

print("Testing data size ",mnist.test.num_examples)


x = tf.placeholder("float",[None, 784])   # ռλ��  ������������MNistͼ�� ÿ��չƽΪ784ά������ ʹ��2ά������������ʾ��Щͼ Nonw��һ��ά�����ⳤ
w= tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10])) # ��������

y = tf.nn.softmax(tf.matmul(x,w)+b) # ʵ�����
y_ = tf.placeholder("float", [None, 10]) # ��ǩֵ
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # ʹ���ݶ��½�����ѧϰ����0.01��С��������  BP

# �������ú�ģ�� ��ʹ��ǰ��Ҫ��ʼ�������ı���
init = tf.initialize_all_variables()

# ʹ��session����ģ��
sess = tf.Session()
sess.run(init)

# ��ʼѵ��ģ��
for i in range(1000):
    batch_xs, batch_ys = mnist.trian.next_batch(100)  
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    
# ����ģ��
# 1���ҳ�Ԥ����ȷ�ı�ǩ
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_ , 1))  # �����һ��bool [true,false, true, true]
# 2�� ���õ���bool -> [1,0,1,1] ȡƽ��ֵ�õ�0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 3�� ���õ���ȷ��
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))