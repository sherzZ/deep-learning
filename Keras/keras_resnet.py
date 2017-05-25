
'''
我们把输入输出大小相同的模块称为identity_block，而把输出比输入小的模块称为conv_block

'''

from keras.layers import merge
from keras.layers.convolutional import Convolution1D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers.core import Dense, Activation,Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input 

# identity_block
