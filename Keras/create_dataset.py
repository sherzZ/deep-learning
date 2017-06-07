#coding=utf-8
import numpy as np
import cv2
import os

from skimage.feature import local_binary_pattern
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

import pickle
# settings for LBP
radius = 3
n_points = 8 * radius

PATH = os.getcwd()

data_path = PATH+'/data/train'
data_dir_list = os.listdir(data_path)


for data in data_dir_list:
    print(data)
    
img_rows =127
img_col = 127
num_channel = 1
num_epoch = 20

# Define the number of classes
num_classes = 2

img_data_list=[]

for dataset in data_dir_list:
    img_list = os.listdir(data_path+'/'+dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        input_img = cv2.imread(data_path+'/'+dataset+'/'+img,0)        
        #cv2.imshow('image', input_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        if input_img is not None:
            #print(input_img.shape)            
            lbp = local_binary_pattern(input_img, n_points, radius) # 提取LBS纹理特征
            max_bins = int(lbp.max() + 1);
            img_hist,_ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));
            #print(input_img.shape)
            input_img_resize = cv2.resize(img_hist, (img_rows,img_col))
            #print(input_img_resize.shape)
            img_data_list.append(input_img_resize)       
    print(len(img_data_list))
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255

print(img_data.shape)
img_data = np.expand_dims(img_data, axis=4) # 因为用的tensorflow(num of samples, img_size,img_size, channelss)
print(img_data.shape)


USE_SKLEARN_PREPROCESSING=False

if USE_SKLEARN_PREPROCESSING:
    # using sklearn for preprocessing
    from sklearn import preprocessing

    def image_to_feature_vector(image, size=(128, 128)):
        # resize the image to a fixed size, then flatten the image into
        # a list of raw pixel intensities 原始像素列表
        return cv2.resize(image, size).flatten()

    img_data_list=[]
    for dataset in data_dir_list:
        img_list=os.listdir(data_path+'/'+ dataset)
        print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
            input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            input_img_flatten=image_to_feature_vector(input_img,(128,128))
            img_data_list.append(input_img_flatten)

    img_data = np.array(img_data_list)
    img_data = img_data.astype('float32')
    print (img_data.shape)
    img_data_scaled = preprocessing.scale(img_data) # 岁数据进行标准化
    print (img_data_scaled.shape)

    print (np.mean(img_data_scaled)) # 均值
    print (np.std(img_data_scaled))  # 方差

    print (img_data_scaled.mean(axis=0))
    print (img_data_scaled.std(axis=0))

    if K.image_dim_ordering()=='th':
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
        print (img_data_scaled.shape)

    else:
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
        print (img_data_scaled.shape)


    if K.image_dim_ordering()=='th':
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],num_channel,img_rows,img_cols)
        print (img_data_scaled.shape)

    else:
        img_data_scaled=img_data_scaled.reshape(img_data.shape[0],img_rows,img_cols,num_channel)
        print (img_data_scaled.shape)

if USE_SKLEARN_PREPROCESSING:
    img_data=img_data_scaled
    
# save data

file = open('train_data_hist.dat','wb')
pickle.dump(img_data, file)
file.close()

#read = open('train_data.dat','rb')
#read_data = pickle.load(read)
#print(read_data.shape)
# label
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,), dtype='int64')

# 两类
labels[0:1000]=0
labels[1000:2000] = 1

names = ['normal','prono']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)
print(Y)
print(Y.shape)

file = open('train_label.dat','wb')
pickle.dump(Y,file)
file.close()

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

