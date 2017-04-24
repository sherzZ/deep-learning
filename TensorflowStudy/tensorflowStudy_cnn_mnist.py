import tensorflow as tf
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

x = tf.placeholder("float",[None, 784])   # 占位符  输入任意数量MNist图像 每张展平为784维的向量 使用2维浮点数张量表示这些图 Nonw第一个维度任意长
w= tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10])) # 行向量？

y = tf.nn.softmax(tf.matmul(x,w)+b) # 实际输出
y_ = tf.placeholder("float", [None, 10]) # 标签值
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) # 使用梯度下降法，学习速率0.01最小化交叉熵  BP

# 以上设置好模型 在使用前需要初始化创建的变量
init = tf.initialize_all_variables()

# 使用session启动模型
sess = tf.Session()
sess.run(init)

# 开始训练模型
for i in range(1000):
    batch_xs, batch_ys = mnist.trian.next_batch(100)  
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    
# 评估模型
# 1、找出预测正确的标签
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_ , 1))  # 会给出一个bool [true,false, true, true]
# 2、 将得到的bool -> [1,0,1,1] 取平均值得到0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# 3、 最后得到正确率
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))