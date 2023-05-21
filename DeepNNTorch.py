"""DNN and CNN Generator."""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class CNNt(nn.Module):
    '''
    CNN Class
    '''
    def __init__(self, conv_channels=[], conv_kernel=[], conv_stride=[], conv_padding=[], conv_pool_kernel=[], conv_pool_stride=[], conv_out=[], out_class=2):
        super(CNNt, self).__init__()
        
        #CNN
        self.cnn = nn.Sequential()
        
        #convolutional layer
        for layer in range(len(conv_kernel)):
            layer_name = 'conv'+str(layer+1)
            self.cnn.add_module(layer_name, self._conv_block(conv_channels[layer], conv_channels[layer+1], kernel_size=conv_kernel[layer], stride=conv_stride[layer], padding=conv_padding[layer]))
            
            layer_name = 'maxpool'+str(layer+1)
            self.cnn.add_module(layer_name, self._max_pooling(conv_pool_kernel[layer], conv_pool_stride[layer]))
            
        
        #final layer, fully connected
        self.fcn = nn.Sequential()
        
        fcn_arch = [conv_out[0]*conv_out[1]*conv_channels[-1], 128, 64, 32, 2]
        print('fcn architecture', fcn_arch)
        for layer in range(len(fcn_arch[1:])):
            layer_name = 'fcn'+str(layer+1)
            if layer < len(fcn_arch[1:])-1:
                self.fcn.add_module(layer_name, self._dense_block(fcn_arch[layer], fcn_arch[layer+1]))
            else:
                self.fcn.add_module(layer_name, self._dense_block_out(fcn_arch[layer], fcn_arch[layer+1]))

    def _conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        return block
    
    def _dense_block(self, in_features, out_features):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.ReLU()
        )
        return block
    
    def _dense_block_out(self, in_features, out_features):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features)
        )
        return block

    def _max_pooling(self, kernel_size, stride_size):
        maxpool = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride_size) #stride – the stride of the window. Default value is kernel_size
        return maxpool

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        x = self.fcn(x)
        return x


class DNNt(nn.Module):
    '''
    DNN Class
    '''
    def __init__(self, fcn_arch=[]):
        super(DNNt, self).__init__()
        
        #DNN
        self.dnn = nn.Sequential()
        
        # fully connected layer
        print('fcn architecture', fcn_arch)
        for layer in range(len(fcn_arch[1:])):
            layer_name = 'fcn'+str(layer+1)
            if layer < len(fcn_arch[1:])-1:
                self.dnn.add_module(layer_name, self._dense_block(fcn_arch[layer], fcn_arch[layer+1]))
            else:
                self.dnn.add_module(layer_name, self._dense_block_out(fcn_arch[layer], fcn_arch[layer+1]))
    
    def _dense_block(self, in_features, out_features):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.ReLU()
        )
        return block
    
    def _dense_block_out(self, in_features, out_features):
        block = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features)
        )
        return block

    def forward(self, x):
        x = self.dnn(x)
        return x


#get weekday
def get_weekday(date_):
    week_day = 'NA'
    try:
        day = int(date_.split("/")[0])
        month = int(date_.split("/")[1])
        year = int(date_.split("/")[2])
        week_day = datetime.date(year,month,day).weekday()
        week_day = str(week_day)
    except:
        pass
    return week_day


# one hot encoding
def one_hot(m, i):
    ohe = list(np.zeros(len(m)))
    ohe[m.index(i)] = 1.0
    return ohe


# metrics
def pred_metrics(gt, pred):
    tp = np.sum((gt == 1) & (pred == 1))
    tn = np.sum((gt == 0) & (pred == 0))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    return tp, tn, fp, fn, tpr, fpr
