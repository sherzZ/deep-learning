#coding=utf-8

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist 数据
# one_hot向量 one-hot 向量是将对应于实际类的的元素为设为 1，其它元素为 0
mnist = input_data.read_data_sets('../../datasets/MNIST_data', one_hot = True)

print("Training data size ",mnist.train.num_examples)

print("Validating data size ",mnist.validation.num_examples)

print("Testing data size ",mnist.test.num_examples)


# 卷积和池化

# 权重初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 步长为1
def conv2d(x, W):
    return tf.nn.conv2d(x, W , strides=[1,1,1,1], padding= 'SAME' )

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 第一层卷积
W_conv1 = weight_variable([5,5,1,32])  # 卷积核大小5*5， 输入通道为1， 输出通道数为32（即 有多少个filter）
b_conv1 = bias_variable([32])

# 为了使用该层 需要将输入x 变成一个4d向量， 第2、3维对应图片的宽和高，最后一维维颜色通道数，灰度图为1， rgb为3

x_image=tf.reshape(x, [-1, 28,28,1])

# 我们把x_image和权值向量进行卷积加上偏置 应用ReLU激活函数， 最后进行max pooling

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 全连接层  现在图片尺寸大小 7*7 共64个， 加入一个1024个神经元的全连接层， 用于处理图片。
# 把池化层输出的张量 reshape成一维向量 flatten 乘上权重矩阵加上偏置然后使用ReLU

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 使用dropout 减少过拟合。 使用一个placeholder来代表一个神经元的输出在dropout中保持不变得概率。
# 这样可以在训练中启用dropout,在测试过程中关闭dropout
# tensorflow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出的scale
# 所以用时 可以不用考虑scale

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层， 添加一个softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 进行训练评估 使用交叉熵 y_ 标签值也是实际值
# argmax 函数，返回tem=nsor对象在某一维度上的其数据最大值所在的索引， 由于标签值是0,1组成，因此最大值1所在的索引就是类别标签
cross_entropy = -tf.reduce_sum(y_ * tf.log( y_conv))
train_step = tf.train.AdamOptimizer(le-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1),tf.argmax(y_ , 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100==0:
        train_accuracy = accuracy.eval(feed_dic = {x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accurcy: %g" %(i, train_accuracy))
        
print("test accurcy %g"%accuracy.eval(feed_dict = {x:mnist.test.image, y_:mnist.test.labels, keep_prob:1.0}))