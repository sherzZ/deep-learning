#coding=utf-8

# 使用caffe生成prototext


import caffe


def create_net(lmdb, mean_file, batch_size, include_acc=False):
    #网络规范
    net = caffe.NetSpec()
    #第一层Data层
    net.data, net.label = caffe.layers.Data(source=lmdb, backend=caffe.params.Data.LMDB, batch_size=batch_size, ntop=2,
                                            transform_param = dict(crop_size = 40, mean_file=mean_file, mirror=True))
    #第二层Convolution视觉层
    net.conv1 = caffe.layers.Convolution(net.data, num_output=20, kernel_size=5,weight_filler={"type": "xavier"},
                                    bias_filler={"type": "constant"})
    #第三层ReLU激活层
    net.relu1 = caffe.layers.ReLU(net.conv1, in_place=True)
    #第四层Pooling池化层
    net.pool1 = caffe.layers.Pooling(net.relu1, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)

    net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size=3, stride=1,num_output=32, pad=1,weight_filler=dict(type='xavier'))
    net.relu2 = caffe.layers.ReLU(net.conv2, in_place=True)
    net.pool2 = caffe.layers.Pooling(net.relu2, pool=caffe.params.Pooling.MAX, kernel_size=3, stride=2)
    #全连层
    net.fc3 = caffe.layers.InnerProduct(net.pool2, num_output=1024,weight_filler=dict(type='xavier'))
    net.relu3 = caffe.layers.ReLU(net.fc3, in_place=True)
    #创建一个dropout层
    net.drop3 = caffe.layers.Dropout(net.relu3, in_place=True)
    net.fc4 = caffe.layers.InnerProduct(net.drop3, num_output=10,weight_filler=dict(type='xavier'))
    #创建一个softmax层
    net.loss = caffe.layers.SoftmaxWithLoss(net.fc4, net.label)
    #训练的prototxt文件不包括Accuracy层,测试的时候需要。
    if include_acc:
        net.acc = caffe.layers.Accuracy(net.fc4, net.label)
        return str(net.to_proto())

    return str(net.to_proto())

def write_net():
    caffe_root = "E:\\wingIde\\PaperCNN\\"    #my-caffe-project目录
    train_lmdb = caffe_root + "train.lmdb"                            #train.lmdb文件的位置
    train_lmdb = caffe_root + "test.lmdb"                            #test.lmdb文件的位置
    mean_file = caffe_root + "mean.binaryproto"                     #均值文件的位置
    train_proto = caffe_root + "train.prototxt"                        #保存train_prototxt文件的位置
    test_proto = caffe_root + "test.prototxt"                        #保存test_prototxt文件的位置
    #写入prototxt文件
    with open(train_proto, 'w') as f:
        f.write(str(create_net(train_lmdb, mean_file, batch_size=64)))

    #写入prototxt文件
    with open(test_proto, 'w') as f:
        f.write(str(create_net(test_proto, mean_file, batch_size=32, include_acc=True)))

if __name__ == '__main__':
    write_net()