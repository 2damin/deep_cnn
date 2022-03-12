from cmath import tan
from re import I
from function.convolution import Conv
from function.pooling import Pool
from function.fc import FC
from function.activation import relu, leaky_relu, sigmoid, tanh
import numpy as np
import torch
import torch.nn as nn
import os,sys
import time
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def convolution():
    print("convolution")
    
    #define the shape of input&weight
    in_w = 3
    in_h = 3
    in_c = 1
    out_c = 16
    batch = 1
    
    X = np.arange(9, dtype=np.float32).reshape([batch,in_c,in_h,in_w]) 
    W = np.array(np.random.standard_normal([out_c,in_c,in_h,in_w]), dtype=np.float32)
    #print(W)
    x_shape = X.shape
    w_shape = W.shape
    Convolution = Conv(batch = x_shape[0],
                       in_c = x_shape[1],
                       out_c = w_shape[0],
                       in_h = x_shape[2],
                       in_w = x_shape[3],
                       k_h = w_shape[2],
                       k_w = w_shape[3],
                       dilation=1,
                       stride=1,
                       pad = 0)

    conv_time =time.time()
    for i in range(5):
        L1 = Convolution.conv(X,W)
    print("conv_time : ",time.time() - conv_time)
    print(L1)
    
    mm_time = time.time()
    for i in range(5):
        L1_a = Convolution.gemm(X,W)
    print("mm_time : ", time.time() - mm_time)
    print(L1_a)
    
    #Convolution using Pytorch
    torch_conv = nn.Conv2d(in_c,out_c,kernel_size=3, stride=1, padding=0, bias=False, dtype=torch.float32)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(W))

    C_torch = torch_conv(torch.tensor(X, requires_grad=False, dtype=torch.float32))
    print("pytorch conv result : ")
    print(C_torch)    
    
def forward_network():
    print("main")

    X = np.arange(108, dtype=np.float64).reshape([1,3,6,6])
    Y = np.array([1], dtype=np.float64)
    W1 = np.random.standard_normal([1,3,3,3])

    #print(W)
    x_shape = X.shape
    w_shape = W1.shape
    Convolution = Conv(batch = x_shape[0],
                       in_c = x_shape[1],
                       out_c = w_shape[0],
                       in_h = x_shape[2],
                       in_w = x_shape[3],
                       k_h = w_shape[2], 
                       k_w = w_shape[3], 
                       dilation=1, 
                       stride=1, 
                       pad = 0)
    
    #layer1_conv
    L1= Convolution.gemm(X,W1)
    print(L1)
    print(L1.shape)
    
    #layer1_pool
    Pooling = Pool(batch = L1.shape[0],
                   in_c = L1.shape[1],
                   out_c = L1.shape[1],
                   in_h = L1.shape[2],
                   in_w = L1.shape[3],
                   kernel = 2,
                   dilation=1,
                   stride=2,
                   pad = 0)
    L1_MAX = Pooling.pool(L1)
    print(L1_MAX)
    
    #layer2_fc
    W2 = np.random.standard_normal([1, L1_MAX.shape[1] * L1_MAX.shape[2] * L1_MAX.shape[3]])
    fc = FC(batch = L1_MAX.shape[0],
            in_c = L1_MAX.shape[1],
            out_c = 1,
            in_h = L1_MAX.shape[2],
            in_w = L1_MAX.shape[3])
    L2 = fc.fc(L1_MAX, W2)
    
    print(L2.shape)
    print(L2)

def shallow_network():
    X = [np.random.standard_normal([1,1,6,6]),
         np.random.standard_normal([1,1,6,6])]
    Y = np.array([1,1], dtype=np.float64)
    # X = [np.random.standard_normal([1,1,6,6])]
    # Y = np.array([1], dtype=np.float64) 
    W1 = np.random.standard_normal([1,1,3,3])
    W2 = np.random.standard_normal([1,4])
    
    L1_h= (X[0].shape[2] + 2 * 0 - W1.shape[3]) // 1 + 1
    L1_w= (X[0].shape[3] + 2 * 0 - W1.shape[3]) // 1 + 1
    
    Convolution = Conv(batch = X[0].shape[0],
                       in_c = X[0].shape[1],
                       out_c = W1.shape[0],
                       in_h = X[0].shape[2],
                       in_w = X[0].shape[3], 
                       k_h = W1.shape[2],
                       k_w = W1.shape[3], 
                       dilation=1,
                       stride=1,
                       pad = 0)
    Conv_diff = Conv(batch = X[0].shape[0],
                     in_c = X[0].shape[1],
                     out_c = W1.shape[0],
                     in_h = X[0].shape[2],
                     in_w = X[0].shape[3],
                     k_h = L1_h,
                     k_w = L1_w,
                     dilation=1,
                     stride=1,
                     pad = 0)
    Fc = FC(batch = X[0].shape[0],
            in_c = X[1].shape[1],
            out_c = 1,
            in_h = L1_h,
            in_w = L1_w)
    Pooling = Pool(batch = X[0].shape[0],
                   in_c = W1.shape[1],
                   out_c = W1.shape[1],
                   in_h = L1_h,
                   in_w = L1_w,
                   kernel = 2,
                   dilation=1,
                   stride=2,
                   pad = 0)
    
    num_epoch = 1000
    for e in range(num_epoch):
        total_loss = 0
        for i in range(len(X)):
            #====Feed forward====
            L1 = Convolution.gemm(X[i],W1)

            L1_act = sigmoid(L1)
            
            L1_max = Pooling.pool(L1_act)
                
            L1_max_flatten = np.reshape(L1_max, (1,-1))

            L2 = Fc.fc(L1_max_flatten, W2)
            
            L2_act = sigmoid(L2)

            loss = np.square(Y[i] - L2_act) * 0.5
            total_loss += loss.item()
            
            #====Backpropagation====
            
            #--delta E / delta W2--
            
            #delta E / delta L2_act
            diff_w2_a = L2_act - Y[i]
            #delta L2_act / delta L2
            diff_w2_b = L2_act*(1 - L2_act)
            #delta L2 / delta W2
            diff_w2_c = L1_max
            #delta E / delta W2
            diff_w2 = np.reshape(diff_w2_a * diff_w2_b * diff_w2_c, (1,-1))
            
            #--delta E / delta W1--
            #delta E / delta L2
            diff_w1 = 1
            diff_w1 *= diff_w2_a * diff_w2_b
            #delta L2 / delta L1_pool
            diff_w1 = diff_w1 * np.reshape(W2,(1,1,2,2)).repeat(2,axis=2).repeat(2,axis=3)
            #delta L1_pool / delta L1_act
            diff_w1 *= np.equal(L1_act, L1_max.repeat(2,axis=2).repeat(2,axis=3)).astype(int)
            #delta L1_act / diff L1
            diff_w1 *= L1_act * (1 - L1_act)

            diff_w1 = Conv_diff.gemm(X[i], np.rot90(np.rot90(diff_w1, axes=(2,3)), axes=(2,3)))
            
            #Update weights
            W2 = W2 - 0.01 * diff_w2 
            W1 = W1 - 0.01 * diff_w1

        if len(X) != 0:
            print("{} epoch - loss : {}".format(e, total_loss / len(X)))

def plot_activation():
    x = np.arange(-10,10,1)
    
    out_relu = relu(x)
    out_leaky = leaky_relu(x)
    out_tanh = tanh(x)
    out_sigmoid = sigmoid(x)

    plt.plot(x, out_relu, 'r', label='relu')
    plt.plot(x, out_leaky, 'b', label='leaky')
    plt.plot(x, out_tanh, 'g', label = 'tanh')
    plt.plot(x, out_sigmoid, 'bs', label='sigmoid')
    plt.ylim([-2,2])
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    #convolution()
    
    #plot_activation()
    
    #forward_network()
    
    shallow_network()