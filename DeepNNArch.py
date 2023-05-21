"""CNN PyTorch Architecture"""
import numpy as np

class CNNarchitecture():
    '''
    CNN Architecture class
    '''
    def __init__(self, kernel=[(2,2)], stride=[(1,1)], padding=[(0,0)], kernel_pool=[(2,2)], stride_pool=[(1,1)]):
        self.kernel=kernel
        self.stride=stride
        self.padding=padding
        self.kernel_pool=kernel_pool
        self.stride_pool=stride_pool
        

    @staticmethod
    def _conv_step_torch(zdim, kernel=2, dilation=1, stride=1, padding=0):
        zdim_out = np.floor( ( (zdim + 2*padding - dilation*(kernel-1) - 1) / stride) + 1)
        return int(zdim_out)
    @staticmethod
    def _deconv_step_torch(zdim, kernel=2, dilation=1, stride=1, padding=0, padding_out=0):
        zdim_out = np.floor( (zdim-1)*stride - 2*padding + dilation*(kernel-1) + padding_out + 1)
        return int(zdim_out)
                
    def conv_architecture_torch(self, image_input, conv_kernel, conv_stride, conv_padding, conv_dilation,
                                pool_kernel, pool_stride, pool_padding, pool_dilation,
                                pooling=False):
        layers = len(conv_kernel[0])
        iii = image_input.shape[0]
        jjj = image_input.shape[1]
        for layer in range(layers):
            print(layer, "input layer", iii, jjj)
            iii = self._conv_step_torch(iii, kernel=conv_kernel[0][layer], dilation=conv_dilation[0][layer], stride=conv_stride[0][layer], padding=conv_padding[0][layer])
            jjj = self._conv_step_torch(jjj, kernel=conv_kernel[1][layer], dilation=conv_dilation[1][layer], stride=conv_stride[1][layer], padding=conv_padding[1][layer])
            print(layer+1, "conv layer", iii, jjj)
            if pooling:
                iii = self._conv_step_torch(iii, kernel=pool_kernel[0][layer], dilation=pool_dilation[0][layer], stride=pool_stride[0][layer], padding=0)
                jjj = self._conv_step_torch(jjj, kernel=pool_kernel[1][layer], dilation=pool_dilation[1][layer], stride=pool_stride[1][layer], padding=0)
                print(layer+1, "pooling layer", iii, jjj)
            if (iii <= 5) | (jjj <= 5):
                break
        return [iii, jjj]
                
    def deconv_architecture_torch(self, image_input, conv_kernel, conv_stride, conv_padding, conv_dilation,
                                  pool_kernel, pool_stride, pool_padding, pool_dilation,
                                  pooling=False):
        layers = len(conv_kernel[0])
        iii = image_input.shape[0]
        jjj = image_input.shape[1]
        for layer in range(layers):
            print(layer, "input layer", iii, jjj)
            iii = self._deconv_step_torch(iii, kernel=conv_kernel[0][layer], dilation=conv_dilation[0][layer], stride=conv_stride[0][layer], padding=conv_padding[0][layer])
            jjj = self._deconv_step_torch(jjj, kernel=conv_kernel[1][layer], dilation=conv_dilation[1][layer], stride=conv_stride[1][layer], padding=conv_padding[1][layer])
            print(layer+1, "deconv layer", iii, jjj)
            if pooling:
                iii = self._deconv_step_torch(iii, kernel=pool_kernel[0][layer], dilation=pool_dilation[0][layer], stride=pool_stride[0][layer], padding=0)
                jjj = self._deconv_step_torch(jjj, kernel=pool_kernel[1][layer], dilation=pool_dilation[1][layer], stride=pool_stride[1][layer], padding=0)
                print(layer+1, "depooling layer", iii, jjj)
            # if (iii >= image_input.shape[0]) | (jjj >= image_input.shape[1]):
                # break
        return [iii, jjj]
    
