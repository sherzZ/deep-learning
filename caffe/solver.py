#coding=utf-8

#net：指定配置文件，cifar10_quick_solver.prototx文件中指定的prototxt文件为examples/cifar10/cifar10_quick_train_test.prototxt，可以使用train_net和test_net分别指定。

#test_iter：测试迭代数。例如：有10000个测试样本，batch_size设为32，那么就需要迭代 10000/32=313次才完整地测试完一次，所以设置test_iter为313。
#test_interval：每训练迭代test_interval次进行一次测试，例如50000个训练样本，batch_size为64，
#             那么需要50000/64=782次才处理完一次全部训练样本，记作1 epoch。所以test_interval设置为782，
#             即处理完一次所有的训练数据后，才去进行测试。

#base_lr：基础学习率，学习策略使用的参数。
#momentum：动量。
#weight_decay：权重衰减。
#lr_policy：学习策略。可选参数：fixed、step、exp、inv、multistep。

#lr_prolicy参数说明：

#fixed: 保持base_lr不变；
#step: step: 如果设置为step，则需要设置一个stepsize，返回base_lr * gamma ^ (floor(iter / stepsize))，其中iter表示当前的迭代次数；
#exp: 返回base_lr * gamma ^ iter，iter为当前的迭代次数；
#inv: 如何设置为inv，还需要设置一个power，返回base_lr * (1 + gamma * iter) ^ (- power)；
#multistep: 如果设置为multistep，则还需要设置一个stepvalue，这个参数和step相似，
#   step是均匀等间隔变化，而multistep则是根据stepvalue值变化；

#stepvalue参数说明： 
#poly: 学习率进行多项式误差，返回base_lr (1 - iter/max_iter) ^ (power)； 
#sigmoid: 学习率进行sigmod衰减，返回base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))。
#display：每迭代display次显示结果。
#max_iter：最大迭代数，如果想训练100 epoch，则需要设置max_iter为100*test_intervel=78200。
#snapshot：保存临时模型的迭代数。
#snapshot_format：临时模型的保存格式。有两种选择：HDF5 和BINARYPROTO ，默认为BINARYPROTO
#snapshot_prefix：模型前缀，就是训练好生成model的名字。不加前缀为iter_迭代数.caffemodel，加之后为lenet_iter_迭代次数.caffemodel。
#solver_mode：优化模式。可以使用GPU或者CPU。

import caffe                                                     #导入caffe包

def write_sovler():
    my_project_root = "E:\\wingIde\\PaperCNN\\"        #my-caffe-project目录
    sovler_string = caffe.proto.caffe_pb2.SolverParameter()                    #sovler存储
    solver_file = my_project_root + 'solver.prototxt'                        #sovler文件保存位置
    sovler_string.train_net = my_project_root + 'train.prototxt'            #train.prototxt位置指定
    sovler_string.test_net.append(my_project_root + 'test.prototxt')         #test.prototxt位置指定
    sovler_string.test_iter.append(100)                                        #测试迭代次数
    sovler_string.test_interval = 500                                        #每训练迭代test_interval次进行一次测试
    sovler_string.base_lr = 0.001                                            #基础学习率   
    sovler_string.momentum = 0.9                                            #动量
    sovler_string.weight_decay = 0.004                                        #权重衰减
    sovler_string.lr_policy = 'fixed'                                        #学习策略           
    sovler_string.display = 100                                                #每迭代display次显示结果
    sovler_string.max_iter = 4000                                            #最大迭代数
    sovler_string.snapshot = 4000                                             #保存临时模型的迭代数
    sovler_string.snapshot_format = 0                                        #临时模型的保存格式,0代表HDF5,1代表BINARYPROTO
    sovler_string.snapshot_prefix = 'examples/cifar10/cifar10_quick'        #模型前缀
    sovler_string.solver_mode = caffe.proto.caffe_pb2.SolverParameter.CPU    #优化模式

    with open(solver_file, 'w') as f:
        f.write(str(sovler_string))   

if __name__ == '__main__':
    write_sovler()