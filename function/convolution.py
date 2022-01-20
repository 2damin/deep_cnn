from stringprep import in_table_c11
import numpy as np

class Conv:
    def __init__(self, batch, in_c, in_w, in_h, out_c, k_h, k_w, dilation, stride, pad):
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
        
    def im2col(self, input):
        mat = np.zeros((self.in_c * self.k_h * self.k_w, self.out_w * self.out_h), dtype=np.int64)
        #channel_size = self.in_h * self.in_w
        for c in range(self.in_c):
            for kh in range(self.k_h):
                for kw in range(self.k_w):
                    input_j = -self.pad + kh * self.dilation
                    for oh in range(self.out_h):
                        if self.check_range(input_j, self.in_h) == False:
                            for ow in range(self.out_w):
                                mat[self.mat_j, self.mat_i] = 0
                                self.mat_i += 1
                        else:
                            input_i = -self.pad + kw * self.dilation
                            for ow in range(self.out_w):
                                if self.check_range(input_i, self.in_h) == False:
                                    mat[self.mat_j, self.mat_i] = 0
                                    self.mat_i += 1
                                else:
                                    mat[self.mat_j, self.mat_i] = input[0, c, input_j, input_i]
                                    self.mat_i += 1
                                input_i += self.stride
                        input_j += self.stride
                    self.mat_i = 0
                    self.mat_j += 1
        return mat
    
    def mm(self, A, B):
        return np.matmul(B, A)
    
    
    def conv(self, A, B):
        C = np.zeros((self.batch, self.out_c, self.out_h, self.out_w), dtype=np.float64)
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
                                
                                
        
    
                        
                            
                            
                            
        
        
        