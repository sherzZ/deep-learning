#coding=utf-8
#加载必要的库

import numpy as np
import sys,os
import caffe
import time
t1 = time.time()

basePath='E:\\wingIde\\PaperCNN\\model_alexNet\\'
net_file= basePath+'deploy.prototxt'
caffe_model= basePath+'trainResult_iter_1200.caffemodel'
mean_file='E:\\wingIde\\PaperCNN\\mean\\mean.npy'

net = caffe.Net(net_file,caffe_model,caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 257) 
transformer.set_channel_swap('data', (2,1,0))

im=caffe.io.load_image(basePath+'1.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data',im)
out = net.forward()

print out
imagenet_labels_filename = basePath + 'words.txt'
labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
output_prob=out['prob']
print 'predicted class is',output_prob.argmax()
t2 = time.time()
print t2-t1
#top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
#for i in np.arange(top_k.size):
    #print top_k[i], labels[top_k[i]]