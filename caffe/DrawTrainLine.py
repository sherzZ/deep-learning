#coding=utf-8
import matplotlib.pyplot as plt    
import numpy as np  
import caffe     
#caffe.set_device(0)    
caffe.set_mode_cpu()     
# 使用SGDSolver，即随机梯度下降算法    
#  下面这里用绝对路径。用～/caffe等相对路径，我这里报出了“Kernel died. Restarting”的错误  
solver = caffe.SGDSolver('E:\\wingIde\\PaperCNN\model_alexNet\\solver.prototxt')    
    
# 等价于solver文件中的max_iter，即最大解算次数    
niter = 1200
# 每隔100次收集一次数据    
display= 100    
    
# 每次测试进行100次解算，10000/100    
test_iter = 2    
# 每500次训练进行一次测试（100次解算），60000/64    
test_interval =10  
    
#初始化   
train_loss = np.zeros(np.ceil(niter * 1.0 / display))     
test_loss = np.zeros(np.ceil(niter * 1.0 / test_interval))    
test_acc = np.zeros(np.ceil(niter * 1.0 / test_interval))    
    
# iteration 0，不计入    
solver.step(1)    
    
# 辅助变量    
_train_loss = 0; _test_loss = 0; _accuracy = 0    
# 进行解算    
for it in range(niter):    
    # 进行一次解算    
    solver.step(1)    
    # 每迭代一次，训练batch_size张图片  “loss”为名字而不是类型  
    _train_loss += solver.net.blobs['loss'].data    
    if it % display == 0:    
        # 计算平均train loss    
        train_loss[it // display] = _train_loss / display    
        _train_loss = 0    
    
    if it % test_interval == 0:    
        for test_it in range(test_iter):    
            # 进行一次测试    
            solver.test_nets[0].forward()    
            # 计算test loss    
            _test_loss += solver.test_nets[0].blobs['loss'].data    
            # 计算test accuracy    
            _accuracy += solver.test_nets[0].blobs['accuracy'].data    
        # 计算平均test loss    
        test_loss[it / test_interval] = _test_loss / test_iter    
        # 计算平均test accuracy    
        test_acc[it / test_interval] = _accuracy / test_iter    
        _test_loss = 0    
        _accuracy = 0    
    
# 绘制train loss、test loss和accuracy曲线    
print '\nplot the train loss and test accuracy\n'    
_, ax1 = plt.subplots()    
ax2 = ax1.twinx()    
    
# train loss -> 绿色    
ax1.plot(display * np.arange(len(train_loss)), train_loss, 'g')    
# test loss -> 黄色    
ax1.plot(test_interval * np.arange(len(test_loss)), test_loss, 'y')    
# test accuracy -> 红色    
ax2.plot(test_interval * np.arange(len(test_acc)), test_acc, 'r')    
    
ax1.set_xlabel('iteration')    
ax1.set_ylabel('loss')    
ax2.set_ylabel('accuracy')    
plt.show()    