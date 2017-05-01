#coding=utf-8
import caffe      

def creat_deploy():
    net = caffe.NetSpec()
    net.conv1 = caffe.layers.Convolution(bottom = 'data', kernel_size = 5, num_output = 20,
                                         weight_filler = dict(type = 'xavier'))
    net.pool1 = caffe.layers.Pooling(net.conv1, kernel_size = 2, stride = 2,
                                     pool = caffe.params.Pooling.MAX)
    net.conv2 = caffe.layers.Convolution(net.pool1, kernel_size = 5, num_output = 50,
                                         weight_filler = dict(type = 'xavier'))
    net.pool2 = caffe.layers.Pooling(net.conv2, kernel_size = 2, stride = 2,
                                     pool = caffe.params.Pooling.MAX)
    net.fc1 =   caffe.layers.InnerProduct(net.pool2, num_output = 500,
                                          weight_filler = dict(type = 'xavier'))
    net.relu1 = caffe.layers.ReLU(net.fc1, in_place = True)
    net.score = caffe.layers.InnerProduct(net.relu1, num_output = 10,
                                          weight_filler = dict(type = 'xavier'))
    net.prob = caffe.layers.Softmax(net.score)
    return net.to_proto()

def write_net(deploy_proto):
    #写入deploy.prototxt文件
    with open(deploy_proto, 'w') as f:
        #写入第一层数据描述
        f.write('input:"data"\n')
        f.write('input_dim:1\n')
        f.write('input_dim:3\n')
        f.write('input_dim:28\n')
        f.write('input_dim:28\n')
        f.write(str(creat_deploy()))

if __name__ == '__main__':
     my_project_root = "/home/Jack-Cui/caffe-master/my-caffe-project/"  
     deploy_proto = my_project_root + "mnist/deploy.prototxt"  
     write_net(deploy_proto)  