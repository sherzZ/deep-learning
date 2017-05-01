#coding=utf-8
import caffe 
import numpy as np
import matplotlib.pyplot as plt


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3) and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data); plt.axis('off')
    plt.show()
    
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatma

caffe.set_mode_cpu()
basePath='E:\\wingIde\\PaperCNN\\model_alexNet\\'
model_def = basePath+'deploy.prototxt'
model_weights = basePath+'trainResult_iter_1200.caffemodel'
mean_file='E:\\wingIde\\PaperCNN\\mean\\mean.npy'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_raw_scale('data', 257) 
transformer.set_channel_swap('data', (2,1,0))

im=caffe.io.load_image(basePath+'test.jpg')
net.blobs['data'].data[...] = transformer.preprocess('data',im)
out = net.forward()
#for layer_name, param in net.params.iteritems():
    #print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)

# the parameters are a list of [weights]
filters = net.params['conv1'][0].data
vis_square(filters.transpose(0, 2, 3, 1))
# The first layer output, conv1 (rectified responses of the filters above, first 96 only)
# conv1 (96, 3, 11, 11) (96,)

# show the first three filters
vis_square(filters[:96].reshape(96*3, 11, 11))

filters_b = net.params['conv1'][1].data
# The first layer output, conv1 (rectified responses of the filters above, first 96 only)
# the params in conv1 is (96, 3, 11, 11) (96,)
print filters_b

# show the output after conv1 layer
# conv1 (1, 96, 55, 55)
feat = net.blobs['conv1'].data[0]
vis_square(feat)

# show the output after pool1 layer
# pool1 (1, 96, 27, 27)
feat = net.blobs['pool1'].data[0]
vis_square(feat)

# show the output after norm1 layer
# norm1 (1, 96, 27, 27)
feat = net.blobs['norm1'].data[0]
vis_square(feat)

# the parameters are a list of weights in conv2 layer
filters = net.params['conv2'][0].data
vis_square(filters[:256].reshape(256*48, 5, 5))

# the parameters are a list of biases.
filters_b = net.params['conv2'][1].data
# vis_square(filters.transpose(0, 2, 3, 1))

# The first layer output, conv1 (rectified responses of the filters above, first 96 only)
print filters_b
#conv2   (256, 48, 5, 5) (256,)

# show the result after conv2
feat = net.blobs['conv2'].data[0]
vis_square(feat)
# conv2 (1, 256, 27, 27)

# show the result after pooling2
feat = net.blobs['pool2'].data[0]
vis_square(feat)
# pool2 (1, 256, 13, 13)

# show the result after LRN 
feat = net.blobs['norm2'].data[0]
vis_square(feat)
# norm2 (1, 256, 13, 13)

# show the result after conv3
feat = net.blobs['conv3'].data[0]
vis_square(feat)
# conv3 (1, 384, 13, 13)

# show the result after conv4
feat = net.blobs['conv4'].data[0]
vis_square(feat)
# conv4 (1, 384, 13, 13)

# show the result after conv5
feat = net.blobs['conv5'].data[0]
vis_square(feat)
# conv5 (1, 256, 13, 13)

# show the result after pooling layer 5
feat = net.blobs['pool5'].data[0]
vis_square(feat)
# pool5 (1, 256, 6, 6)

# show the result after fc6 layer
feat = net.blobs['fc6'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()
# fc6   (1, 4096)

# show the result after fc7
feat = net.blobs['fc7'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()
# fc7   (1, 4096)

# show the result after fc8
feat = net.blobs['fc8'].data[0]
plt.subplot(2, 1, 1)
plt.plot(feat.flat)
plt.subplot(2, 1, 2)
_ = plt.hist(feat.flat[feat.flat > 0], bins=100)
plt.show()
# fc8   (1, 1000)

# show the result after prob layer
feat = net.blobs['prob'].data[0]
plt.figure(figsize=(15, 3))
plt.plot(feat.flat)
plt.show()
# prob  (1, 1000)