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
    print("main")
    
    X = np.arange(49152, dtype=np.float64).reshape([1,3,128,128])
    Y = np.array([1], dtype=np.float64)
    W = np.random.standard_normal([32,3,3,3])
    #print(W)
    x_shape = X.shape
    w_shape = W.shape
    Convolution = Conv(x_shape[0], x_shape[1], x_shape[2], x_shape[3], out_c = w_shape[0], k_h = w_shape[2], k_w = w_shape[3], dilation=1, stride=1, pad = 0)
    
    conv_time =time.time()
    for i in range(10):
        L1 = Convolution.conv(X,W)
    print("conv_time : ",time.time() - conv_time)
    #print(L1)
    
    
    mm_time = time.time()
    for i in range(10):
        L1_a = Convolution.mm(X,W)
    print("mm_time : ", time.time() - mm_time)
    #print(L1_a)
    
    #Convolution using Pytorch
    torch_conv = nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1, dtype=torch.float64)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(W))
    #torch_conv.weight.data.fill_(1)
    C_torch = torch_conv(torch.tensor(X, requires_grad=False))
    print("pytorch conv result : ")
    print(C_torch)
    
    
def main():
    print("main")

    X = np.arange(108, dtype=np.float64).reshape([1,3,6,6])
    Y = np.array([1], dtype=np.float64)
    W1 = np.random.standard_normal([1,3,3,3])

    #print(W)
    x_shape = X.shape
    w_shape = W1.shape
    Convolution = Conv(x_shape[0], x_shape[1], x_shape[2], x_shape[3], out_c = w_shape[0], k_h = w_shape[2], k_w = w_shape[3], dilation=1, stride=1, pad = 0)
    
    #layer1_conv
    L1= Convolution.mm(X,W1)
    print(L1)
    print(L1.shape)
    
    #layer1_pool
    Pooling = Pool(L1.shape[0], L1.shape[1], L1.shape[2], L1.shape[3], out_c = L1.shape[1], kernel = 2, dilation=1, stride=2, pad = 0)
    L1_MAX = Pooling.pool(L1)
    print(L1_MAX)
    
    #layer2_fc
    W2 = np.random.standard_normal([L1_MAX.shape[0], L1_MAX.shape[1] * L1_MAX.shape[2] * L1_MAX.shape[3]])
    fc = FC(L1_MAX.shape[0], L1_MAX.shape[1], L1_MAX.shape[2], L1_MAX.shape[3])
    L2 = fc.fc(L1_MAX, W2)
    
    print(L2.shape)
    print(L2)
    
    act_values = [[],[],[],[]]
    x_values = np.arange(-10,10,1)
    for i in range(-10, 10, 1):
        print(i)
        act_values[0].append(relu(i))
        act_values[1].append(leaky_relu(i))
        act_values[2].append(tanh(i))
        act_values[3].append(sigmoid(i))

    plt.plot(x_values, act_values[0], 'r', x_values, act_values[1], 'b', x_values, act_values[2], 'g', x_values, act_values[3], 'bs')
    plt.show()
        

def shallow_network():
    X = [np.random.standard_normal([1,1,6,6]),
         np.random.standard_normal([1,1,6,6])]
    Y = np.array([1,1], dtype=np.float64)
    # X = [np.random.standard_normal([1,1,6,6])]
    # Y = np.array([1], dtype=np.float64) 
    W1 = np.random.standard_normal([1,1,3,3])
    W2 = np.random.standard_normal([4,1])
    
    L1_h= (X[0].shape[2] + 2 * 0 - W1.shape[3]) // 1 + 1
    L1_w= (X[0].shape[3] + 2 * 0 - W1.shape[3]) // 1 + 1
    
    Convolution = Conv(X[0].shape[0], X[0].shape[1], X[0].shape[2], X[0].shape[3], out_c = W1.shape[0], k_h = W1.shape[2], k_w = W1.shape[3], dilation=1, stride=1, pad = 0)
    Conv_diff = Conv(X[0].shape[0], X[0].shape[1], X[0].shape[2], X[0].shape[3], out_c = W1.shape[0], k_h = L1_h, k_w = L1_w, dilation=1, stride=1, pad = 0)
    Fc = FC(X[0].shape[0], X[1].shape[1], L1_h, L1_w)
    Pooling = Pool(X[0].shape[0], W1.shape[1], L1_h, L1_w, out_c = W1.shape[1], kernel = 2, dilation=1, stride=2, pad = 0)
    
    num_epoch = 100
    for iter in range(num_epoch):
        total_loss = 0
        for i in range(len(X)):
            L1 = Convolution.mm(X[i],W1)
            #print(L1)
            
            L1_act = sigmoid(L1)
            
            L1_max = Pooling.pool(L1_act)
            
            #print(L1_max)
            
            L1_max_flatten = np.reshape(L1_max, (1,-1))
            #print(L1_max_flatten)
            
            L2 = Fc.fc(L1_max_flatten, W2)
            #L2 = np.dot(L1_max_flatten, W2)
            #print(L2)
            
            L2_act = sigmoid(L2)
            #print(L2_act)
            
            loss = np.square(Y[i] - L2_act) * 0.5
            total_loss += loss.item()
            
            #Backpropagation
            
            #--delta E / delta W2--
            
            #delta E / delta L2_act
            diff_w2_a = L2_act - Y[i]
            #delta L2_act / delta L2
            diff_w2_b = L2_act*(1 - L2_act)
            #delta L2 / delta W2
            diff_w2_c = L1_max
            #delta E / delta W2
            diff_w2 = np.reshape(diff_w2_a * diff_w2_b * diff_w2_c, (-1,1))
            
            #--delta E / delta W1--
            #delta E / delta L2
            diff_w1 = 1
            diff_w1 *= diff_w2_a * diff_w2_b
            #diff_w1_a = diff_w2_a
            
            #delta L2 / delta L1_pool
            diff_w1 = diff_w1 * np.reshape(W2,(1,1,2,2)).repeat(2,axis=2).repeat(2,axis=3)
            #diff_w1_b = np.reshape(W2,(1,1,2,2)).repeat(2,axis=2).repeat(2,axis=3)
            
            #delta L1_pool / delta L1_act
            diff_w1 *= np.equal(L1_act, L1_max.repeat(2,axis=2).repeat(2,axis=3)).astype(int)
            #diff_w1_c = np.equal(L1_act, L1_max.repeat(2,axis=2).repeat(2,axis=3)).astype(int)
            
            #delta L1_act / diff L1
            diff_w1 *= L1_act * (1 - L1_act)
            #diff_w1_d = L1_act * (1 - L1_act)

            diff_w1 = Conv_diff.mm(X[i], np.rot90(np.rot90(diff_w1, axes=(2,3)), axes=(2,3)))
            
            #print("diff_w1 :", diff_w1 )
            #print(W1, W2)
            W2 = W2 - 0.01 * diff_w2 
            W1 = W1 - 0.01 * diff_w1
            
            #delta L1 / delta W1
            #diff_w1 = diff_w1[]
            #diff_w1_e = X[i] 
            #print(diff_w1_e)
            #print(np.rot90(diff_w1_c, 1))
            #delta E / delta W1
            #diff_w1 = diff_w1_a * diff_w1_b * diff_w1_c * diff_w1_d * diff_w1_e
            #print(diff_w1)
            #sys.exit(1)
        if len(X) != 0:
            print("loss : ", total_loss / len(X))
    
if __name__ == "__main__":
    #convolution()
    
    #main()
    
    shallow_network()