from stringprep import in_table_c11
import numpy as np

#Convolution layer
class Conv:
    def __init__(self, batch, in_c, out_c, in_h, in_w, k_h, k_w, dilation, stride, pad):
        self.batch = batch
        self.in_c = in_c
        self.in_w = in_w
        self.in_h = in_h
        self.out_c = out_c
        self.k_h = k_h
        self.k_w = k_w
        self.dilation = dilation
        self.stride = stride
        self.pad = pad
        self.out_w = (in_w + 2 * pad - k_w) // stride + 1
        self.out_h = (in_h + 2 * pad - k_h) // stride + 1
        self.mat_i = 0
        self.mat_j = 0
        
    def check_range(self, a, b):
        return a > -1 and a < b

    #IM2COL. Change N-Dim input matrix to 2D matrix 
    def im2col(self, input):
        mat = np.zeros((self.in_c * self.k_h * self.k_w, self.out_w * self.out_h), dtype=np.float32)
        #channel_size = self.in_h * self.in_w
        for c in range(self.in_c):
            for kh in range(self.k_h):
                for kw in range(self.k_w):
                    in_j = kh * self.dilation - self.pad
                    for oh in range(self.out_h):
                        if not self.check_range(in_j, self.in_h):
                            for ow in range(self.out_w):
                                mat[self.mat_j, self.mat_i] = 0
                                self.mat_i += 1
                        else:
                            in_i = kw * self.dilation -self.pad
                            for ow in range(self.out_w):
                                if not self.check_range(in_i, self.in_h):
                                    mat[self.mat_j, self.mat_i] = 0
                                    self.mat_i += 1
                                else:
                                    mat[self.mat_j, self.mat_i] = input[0, c, in_j, in_i]
                                    self.mat_i += 1
                                in_i += self.stride
                        in_j += self.stride
                        self.mat_i = 0
                    self.mat_j += 1
        self.mat_i = 0
        self.mat_j = 0
        return mat
    
    #GEMM. 2D matrix multiplication
    def gemm(self, A, B):
        a_mat = self.im2col(A)
        b_mat = B.reshape(B.shape[0], -1)
        c_mat = np.matmul(b_mat, a_mat)
        c = c_mat.reshape([self.batch, self.out_c, self.out_h, self.out_w])
        return c
    
    #Naive convolution. Sliding window metric
    def conv(self, A, B):
        C = np.zeros((self.batch, self.out_c, self.out_h, self.out_w), dtype=np.float32)
        for b in range(self.batch):
            for oc in range(self.out_c):
                for oh in range(self.out_h):
                    for ow in range(self.out_w):
                        a_j = -self.pad + oh * self.stride
                        for kh in range(self.k_h):
                            if self.check_range(a_j, self.in_h) == False:
                                C[b, oc, oh, ow] += 0
                            else:
                                a_i = -self.pad + ow * self.stride
                                for kw in range(self.k_w):
                                    if self.check_range(a_i, self.in_w) == False:
                                        C[b, oc, oh, ow] += 0
                                    else:
                                        C[b, oc, oh, ow] += np.dot(A[b, :, a_j, a_i],B[oc, :, kh, kw])
                                    a_i += self.stride
                            a_j += self.stride
        return C
                                
                                
        
    
                        
                            
                            
                            
        
        
        