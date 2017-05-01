#coding=utf-8
import numpy as np  
import os  
import sys  
import argparse  
import glob  
import time,datetime  
import scipy.io as scio  
import caffe  
caffe.set_mode_gpu();  

if len(sys.argv)!=3:  
    print "Usage: python convert_mean.py mean.binaryproto mean.npy"  
    sys.exit()  

blob = caffe.proto.caffe_pb2.BlobProto()  
bin_mean = open( sys.argv[1] , 'rb' ).read()  
blob.ParseFromString(bin_mean)  
arr = np.array( caffe.io.blobproto_to_array(blob) )  
npy_mean = arr[0]  
np.save( sys.argv[2] , npy_mean )