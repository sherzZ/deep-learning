#coding=utf-8
import tensorflow as tf

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# Network Parameters
n_input = 224*224 # data input (img shape: 28*28) (修改)
n_classes = 3 # MNIST total classes (0-9 digits)   （三类）
dropout = 0.8 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float") # dropout (keep probability)

# 根据shape定义卷积核 与 偏置
def weight_variable(shape,name):
    init = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.variables(init)

def bais_variable(shape, name):
    init=tf.Variable(tf.random_normal([64]), name=name)
    #init = tf.constant(0.1, shape=shape,name=name)
    return tf.variables(init)

# 卷积运算代码
# l_input 是输入样本，这里即图像。 x的shape=[batch, height,width,channels]
# batch:是输入样本的数量
# height width:每张图像的高和宽
# channels: 输入的通道。 =1 灰度 =3 RGB彩色

# W表示卷积核的参数。 shape=[kernel_h, kernel_w, in_channels, out_channel]

# strides:步长
# padding参数用来控制图片的边距，’SAME’表示卷积后的图片与原图片大小相同，
#’VALID’的话卷积以后图像的高为Height（out）=Height原图−Height卷积核+1/Stride(Height)， 宽也同理。

def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='VALID'),b), name=name)

# 步长与核大小一致
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID', name=name)

# 归一化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


  
# 定义网络
def model(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 224, 224, 1])
    
    # Convolution Layer 第一层
    with tf.name_scope('layer1') as scope:        
        conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
        # Max Pooling (down-sampling)
        pool1 = max_pool('pool1', conv1, k=2)
        # Apply Normalization
        norm1 = norm('norm1', pool1, lsize=4)
        # Apply Dropout
        norm1 = tf.nn.dropout(norm1, _dropout)
        #conv1 image show
        tf.image_summary("conv1", conv1)
        
    # Convolution Layer 第二层
    with tf.name_scope('layer2') as scope:
        conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        pool2 = max_pool('pool2', conv2, k=2)
        # Apply Normalization
        norm2 = norm('norm2', pool2, lsize=4)
        # Apply Dropout
        norm2 = tf.nn.dropout(norm2, _dropout)

    # Convolution Layer 第三层
    with tf.name_scope('layer3') as scope:
        conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
        # Max Pooling (down-sampling)
        pool3 = max_pool('pool3', conv3, k=2)
        # Apply Normalization
        norm3 = norm('norm3', pool3, lsize=4)
        # Apply Dropout
        norm3 = tf.nn.dropout(norm3, _dropout)
        
    #conv4 第四层
    with tf.name_scope('layer2') as scope:
        conv4 = conv2d('conv4', norm3, _weights['wc4'], _biases['bc4'])
        # Max Pooling (down-sampling)
        pool4 = max_pool('pool4', conv4, k=2)
        # Apply Normalization
        norm4 = norm('norm4', pool4, lsize=4)
        # Apply Dropout
        norm4 = tf.nn.dropout(norm4, _dropout)
        
    # Fully connected layer 全连接层
    with tf.name_scope('full_1') as scope:
        dense1 = tf.reshape(norm4, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
        dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1') # Relu activation
    # 第二个全连接层    
    with tf.name_scope('full_1') as scope:
        dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation

    # Output, class prediction
    with tf.name_scope('out_layer') as scope:
        out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out

# Store layers weight & bias
weights = {
    'wc1': weight_variable([3,3,1,64],"weight"),       # 最后一维维输出
    'wc2': weight_variable([3,3,64,128],"weight"),
    'wc3': weight_variable([3,3,128,256],"weight"),
    'wc4': weight_variable([2,2,256,512],"weight"),   #fully connect layer
    'wd1': weight_variable([2*2*512，1024],"weight"),        # 数值要修改
    'wd2': weight_variable([1024，1024],"weight"),
    'out': tf.Variable(tf.random_normal([1024, 3]))  #最终生成3类
}
biases = {
    'bc1': bais_variable([64], "bias"),
    'bc2': bais_variable([128], "bias"),
    'bc3': bais_variable([256], "bias"),
    'bc4': bais_variable([512], "bias"),
    'bd1': bais_variable([1024], "bias"),
    'bd2': bais_variable([1024], "bias"),
    'out': bais_variable([n_classes], "bias"),
}
