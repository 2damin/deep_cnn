import numpy as np

#2D pool layer
class Pool:
    def __init__(self, batch, in_c, out_c, in_h, in_w, kernel, dilation, stride, pad):
        self.batch = batch
        self.in_c = in_c
        self.in_w = in_w
        self.in_h = in_h
        self.out_c = out_c
        self.kernel = kernel
        self.dilation = dilation
        self.stride = stride
        self.pad = pad
        self.out_w = (in_w + 2 * pad - kernel) // stride + 1
        self.out_h = (in_h + 2 * pad - kernel) // stride + 1
        self.mat_i = 0
        self.mat_j = 0
    
    def pool(self, A):
        C = np.zeros([self.batch,self.out_c,self.out_h,self.out_w])
        for b in range(self.batch):
            for c in range(self.in_c):
                for oh in range(self.out_h):
                    aj = -self.pad + oh * self.stride
                    for ow in range(self.out_w):
                        ai = -self.pad + ow * self.stride
                        v = np.amax(A[:,c,aj:aj+self.kernel,ai:ai+self.kernel])
                        C[b,c,oh,ow] = v                        
        return C
                    