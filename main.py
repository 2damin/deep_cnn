from re import I
from function.convolution import Conv
from function.pooling import Pool
from function.fc import FC
import numpy as np
import torch
import torch.nn as nn
import os
import time
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
    
    
    
if __name__ == "__main__":
    #convolution()
    
    main()