#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2

def conv2(X, k):
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    # 有padding 和 strid H_out = (x_row+2*H_padding-k_row)/H-strid +1 
    # W_out = (x_col + 2*W_padding - k_col)/W_strid +1
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    ret = np.empty((ret_row, ret_col))
    for y in range(ret_row):
        for x in range(ret_col):
            sub = X[y : y + k_row, x : x + k_col]
            ret[y,x] = np.sum(sub * k)
    return ret

# randn（d0,d1,d2,...,dn）从标准正太分布返回一个或多个样本值
# d0,..dn表示维度

class ConvLayer:
    def __init__(self, in_channel, out_channel, kernel_size):
        # randn(in_channel, out_channel, kernel_size, kernel_size)
        # 表示一个in_channel行 out-channel列的矩阵，矩阵中每个元素是kernel_size*kernel_size
        # 这样w的每一行in_channel对应一个样本的权值，有多少out_channel列表示有多少个特征
        self.w = np.random.randn(in_channel, out_channel, kernel_size, kernel_size)
        self.b = np.zeros((out_channel))
        
    def _relu(self, x):
        x[x < 0] = 0
        return x
    
    def forward(self, in_data):
        # assume the first index is channel index
        in_channel, in_row, in_col = in_data.shape
        out_channel, kernel_row, kernel_col = self.w.shape[1], self.w.shape[2], self.w.shape[3]
        self.top_val = np.zeros((out_channel, in_row - kernel_row + 1, in_col - kernel_col + 1))
        for j in range(out_channel):
            for i in range(in_channel):
                self.top_val[j] += conv2(in_data[i], self.w[i, j])
            self.top_val[j] += self.b[j]
            self.top_val[j] = self._relu(self.top_val[j])
        return self.top_val

# 测试

# 原图显示
mat = cv2.imread("test.jpg",0)
row, col = mat.shape
in_data = mat.reshape(1, row, col)
in_data = in_data.astype(np.float)/255
plt.imshow(in_data[0], cmap="Greys_r") 
#plt.show()

# 均值滤波  模糊图像作用
meanConv = ConvLayer(1,1,5)
meanConv.w[0,0] = np.ones((5,5))/(5*5)
mean_out = meanConv.forward(in_data)
plt.imshow(mean_out[0], cmap="Greys_r")
#plt.show()

# sobel滤波 纵向梯度的核
sobelConv = ConvLayer(1,1,3)
sobelConv.w[0,0] = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sobel_out = sobelConv.forward(in_data)
plt.imshow(sobel_out[0], cmap='Greys_r')
#plt.show()

# Gabor filter
def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    (y, x) = np.meshgrid(np.arange(-1,2), np.arange(-1,2))
    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

print (gabor_fn(2, 0, 0.3, 0, 2))
gaborConv = ConvLayer(1,1,3)
gaborConv.w[0,0] = gabor_fn(2, 0, 0.3, 1, 2)
gabor_out = gaborConv.forward(in_data)
plt.imshow(gabor_out[0], cmap='Greys_r')
plt.show()

# 下采样层 池化层pooling
class MaxpoolingLayer:
    def __init__(self, kernel_size, name='MaxPool'):
        self.kernel_size = kernel_size
        
    def forward(self, in_data):
        in_batch, in_channel, in_row, in_col = in_data.shape
        k = self.kernel_size
        out_row = in_row / k + (1 if in_row % k != 0 else 0)
        out_col = in_col / k + (1 if in_col % k != 0 else 0)
    
        self.flag = np.zeros_like(in_data)
        ret = np.empty((in_batch, in_channel, out_row, out_col))
        for b_id in range(in_batch):
            for c in range(in_channel):
                for oy in range(out_row):
                    for ox in range(out_col):
                        height = k if (oy + 1) * k <= in_row else in_row - oy * k
                        width = k if (ox + 1) * k <= in_col else in_col - ox * k
                        idx = np.argmax(in_data[b_id, c, oy * k: oy * k + height, ox * k: ox * k + width])
                        offset_r = idx / width
                        offset_c = idx % width
                        self.flag[b_id, c, oy * k + offset_r, ox * k + offset_c] = 1                        
                        ret[b_id, c, oy, ox] = in_data[b_id, c, oy * k + offset_r, ox * k + offset_c]
        return ret  
    
    # 后向传播
    
    def backward(self, residual):
        in_channel, out_channel, kernel_size = self.w.shape
        in_batch = residual.shape[0]
        # gradient_b        
        self.gradient_b = residual.sum(axis=3).sum(axis=2).sum(axis=0) / self.batch_size
        # gradient_w
        self.gradient_w = np.zeros_like(self.w)
        for b_id in range(in_batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    self.gradient_w[i, o] += conv2(self.bottom_val[b_id], residual[o])
        self.gradient_w /= self.batch_size
        # gradient_x
        gradient_x = np.zeros_like(self.bottom_val)
        for b_id in range(in_batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    gradient_x[b_id, i] += conv2(padding(residual, kernel_size - 1), rot180(self.w[i, o]))
        gradient_x /= self.batch_size
        # update
        self.prev_gradient_w = self.prev_gradient_w * self.momentum - self.gradient_w
        self.w += self.lr * self.prev_gradient_w
        self.prev_gradient_b = self.prev_gradient_b * self.momentum - self.gradient_b
        self.b += self.lr * self.prev_gradient_b
        
        
        
        # 下采样层 采样层下一层是卷积层
        # rot180°：旋转：表示对矩阵进行180度旋转（可通过行对称交换和列对称交换完成
        def rot180(in_data):
            ret = in_data.copy()
            yEnd = ret.shape[0] - 1
            xEnd = ret.shape[1] - 1
            for y in range(ret.shape[0] / 2):
                for x in range(ret.shape[1]):
                    ret[yEnd - y][x] = ret[y][x]
            for y in range(ret.shape[0]):
                for x in range(ret.shape[1] / 2):
                    ret[y][xEnd - x] = ret[y][x]
            return ret   
        
        # padding:扩充
        def padding(in_data, size):
            cur_r, cur_w = in_data.shape[0], in_data.shape[1]
            new_r = cur_r + size * 2
            new_w = cur_w + size * 2
            ret = np.zeros((new_r, new_w))
            ret[size:cur_r + size, size:cur_w+size] = in_data
            return ret        