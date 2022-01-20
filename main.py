from re import I
from function.convolution import Conv
import numpy as np
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    print("main")
    
    A = np.arange(75, dtype=np.float64).reshape((1,3,5,5))
    B = np.ones((3, 3, 3, 3), dtype=np.float64)

    #Get shape of A, B
    a_size = A.shape
    b_size = B.shape
    print(A)
    
    #Change weight's shape 4D -> 2D
    B_mat = B.reshape(b_size[0], -1)
    print(B.shape)
    
    #Create Convolution 
    Convolution = Conv(a_size[0], a_size[1], a_size[2], a_size[3], b_size[0], b_size[2], b_size[3], dilation=1, stride=1, pad=1)
    
    A_mat = Convolution.im2col(A)
    
    print("A_mat :")
    print(A_mat)
    print("A_mat shape : ", A_mat.shape)
    
    C_mat = Convolution.mm(A_mat, B_mat)
    
    print("im2col&mm convolution result : ")
    print("C_mat :")
    print(C_mat)
    print("C_mat shape : ", C_mat.shape)
    
    #Convolution using Pytorch
    torch_conv = nn.Conv2d(3,3,kernel_size=3, stride=1, padding=1, dtype=torch.float64)
    #torch_conv.weight = torch.nn.Parameter(torch.tensor(B))
    torch_conv.weight.data.fill_(1)
    C_torch = torch_conv(torch.tensor(A, requires_grad=False))
    print("pytorch conv result : ")
    print(C_torch)
    

    #Convolution naively
    origin_C = Convolution.conv(A, B)
    
    print("naive convolution result :")
    print(origin_C)
    
    
    
    

if __name__ == "__main__":
    main()