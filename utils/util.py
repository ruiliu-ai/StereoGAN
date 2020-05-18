#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import math
import re
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os


flag_check_shape = False
flag_bn = False ## default use bn or not
# True = True 
activefun_default = nn.PReLU(num_parameters=1, init = 0.1) # default activate function

def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def msg_conv(obj_in, obj_conv, obj_out):
    return "\n input: %s\n conv: %s\n output: %s\n" % (str(obj_in.shape), str(obj_conv), str(obj_out.shape))

def msg_shapes(**args):
    n = len(args)
    msg = ""
    shapes = [str(arg.shape) for arg in args]
    for i in range(n-1):
        msg += "%s--->%s\n" % (shapes[i], shapes[i+1])
    return msg

# weight init
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = fanin_init(m.weight.data.size())
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

# corr1d
class Corr1d(nn.Module):
    def __init__(self, kernel_size=1, stride=1, D=1, simfun=None):
        super(Corr1d, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.D = D
        if(simfun is None):
            self.simfun = self.simfun_default
        else: # such as simfun = nn.CosineSimilarity(dim=1)
            self.simfun = simfun
    
    def simfun_default(self, fL, fR):
        return (fL*fR).sum(dim=1);
        
    def forward(self, fL, fR):
        bn, c, h, w = fL.shape
        D = self.D  ## top channels == displacement
        stride = self.stride
        kernel_size = self.kernel_size
        corrmap = torch.zeros(bn, D, h, w).type_as(fL.data)  ## output shape : [ bn , maxdisplacement, h, w ]
        corrmap[:, 0] = self.simfun(fL, fR)
        for i in range(1, D):
            if(i >= w): break
            idx = i*stride
            corrmap[:, i, :, idx:] = self.simfun(fL[:, :, :, idx:], fR[:, :, :, :-idx])
        if(kernel_size>1):
            assert kernel_size%2 == 1
            m = nn.AvgPool2d(kernel_size, stride=1, padding=kernel_size//2)
            corrmap = m(corrmap)
        return corrmap




class Corr1d_x(nn.Module):
    def __init__(self, pad=13, kernel_size=1, max_displacement=13, stride1=1,  stride2=1, pad_shift=-10):
        super(Corr1d_x, self).__init__()
        
        self.pad_size = pad
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.pad_shift = pad_shift
        self.kernel_radius = int( ( kernel_size - 1 ) // 2)
        self.border_size = int(max_displacement + self.kernel_radius)
        self.neighborhood_grid_radius = int(self.max_displacement // self.stride2)
        self.neighborhood_grid_width_ = int(self.neighborhood_grid_radius * 2 + 1)
        self.top_channels = int(self.neighborhood_grid_width_)
        
        
    def forward_slow(self, img1, img2):
        rbot1 = F.pad(img1,  (self.pad_size, self.pad_size))
        rbot2 = F.pad(img2, (self.pad_size - self.pad_shift,   self.pad_size + self.pad_shift))
        num, bchannels, bheight, bwidth = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
        top_width =  int(math.ceil(((bwidth + 2*self.pad_size) - self.border_size*2) / self.stride1))
        top_height = int(math.ceil( ((bheight + 2*self.kernel_radius) - self.kernel_radius * 2) / self.stride1))
        
        top = torch.zeros(num, self.top_channels, top_height, top_width).type_as(img1.data)
        
        x_shift = - self.neighborhood_grid_radius
        
        sumelems = self.kernel_size * self.kernel_size * bchannels
        for y in range(0, top_width):
            for x in range(0, top_height):
                for ch in range(0, self.top_channels):
                    s2o = int(( ch % self.neighborhood_grid_width_ + x_shift ) * self.stride2)
                    
                    patch_y_start = y 
                    patch_y_end = y + self.kernel_size 
                    
                    patch1_x_start = x * self.stride1 + self.max_displacement 
                    patch1_x_end = x * self.stride1 + self.max_displacement + self.kernel_size 
                    patch1 = rbot1[:, :, patch_y_start : patch_y_end  ,  patch1_x_start : patch1_x_end  ]
                    
                    patch2_x_start = patch1_x_start + s2o 
                    patch2_x_end = patch1_x_end + s2o 
                    patch2 = rbot2[:,:, patch_y_start : patch_y_end , patch2_x_start : patch2_x_end ]
                    
                    
#                    print('p1 shape', patch1.shape,  'p2 shape', patch2.shape)
#                    print('p1x : {}-{}, s2o : {}  , p2x: {}-{}, y:{}-{}'.format(patch1_x_start, patch1_x_end, s2o, patch2_x_start, patch2_x_end, patch_y_start, patch_y_end  ))
                    
                    s = torch.sum(patch1.mul(patch2) , (1,2,3)) / sumelems
#                    print('s', s.shape)
                    
                    top[:, ch , y, x] = s 
        return top
    
    def corr_prod_func(self, fL, fR):
        return (fL*fR).sum(dim=1)

    def corr_cos_func(self, fL, fR):
        prod = (fL*fR).sum(dim=1)
        modL = (fL ** 2).sum(dim=1)
        modR = (fR ** 2).sum(dim=1)
        modL = modL ** 0.5
        modR = modR ** 0.5
        return prod / (modL * modR + 1e-8)
    
    def forward(self, img1, img2):
        #self.corr_func = self.corr_cos_func
        self.corr_func = self.corr_prod_func
        num, bchannels, bheight, bwidth = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
        top_width =  int(math.ceil(((bwidth + 2*self.pad_size) - self.border_size*2) / self.stride1))
        top_height = int(math.ceil( ((bheight + 2*self.kernel_radius) - self.kernel_radius * 2) / self.stride1))
        
        top = torch.zeros(num, self.top_channels, top_height, top_width).type_as(img1.data)
        
        x_shift = - self.neighborhood_grid_radius
        
        sumelems = self.kernel_size * self.kernel_size * bchannels
        for ch in range(0,  self.top_channels):
            s2o = int(( ch % self.neighborhood_grid_width_ + x_shift ) * self.stride2) + self.pad_shift
#            print('s2o:  ', s2o)
            if s2o > 0 :
#                print(img1[:, :, :, :-s2o].shape , img2[:, :, :, s2o:].shape )
                top[:,ch,:, :-s2o] = self.corr_func(img1[:, :, :, :-s2o] , img2[:, :, :, s2o:])
            elif s2o < 0:
#                print(img1[:, :, :, -s2o:].shape, img2[:, :, :, :s2o].shape )
                top[:, ch, :, -s2o:] = self.corr_func(img1[:, :, :, -s2o:], img2[:, :, :, :s2o])
            else:
                top[:, ch, :, :]  = self.corr_func(img1, img2)
                
        top = top / sumelems
        return top

class Corr1d_x_group(nn.Module):
    def __init__(self, pad=13, kernel_size=1, max_displacement=13, stride1=1,  stride2=1, pad_shift=-10, num_group=4):
        super(Corr1d_x_group, self).__init__()

        self.pad_size = pad
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.pad_shift = pad_shift
        self.kernel_radius = int( ( kernel_size - 1 ) // 2)
        self.border_size = int(max_displacement + self.kernel_radius)
        self.neighborhood_grid_radius = int(self.max_displacement // self.stride2)
        self.neighborhood_grid_width_ = int(self.neighborhood_grid_radius * 2 + 1)
        self.top_channels = int(self.neighborhood_grid_width_)

        self.num_group = num_group


    def corr_func(self, fL, fR):
        return (fL*fR).sum(dim=1);

    def corr_forward(self, img1, img2):
        num, bchannels, bheight, bwidth = img1.shape[0], img1.shape[1], img1.shape[2], img1.shape[3]
        top_width =  int(math.ceil(((bwidth + 2*self.pad_size) - self.border_size*2) / self.stride1))
        top_height = int(math.ceil( ((bheight + 2*self.kernel_radius) - self.kernel_radius * 2) / self.stride1))

        top = torch.zeros(num, self.top_channels, top_height, top_width).type_as(img1.data)

        x_shift = - self.neighborhood_grid_radius

        sumelems = self.kernel_size * self.kernel_size * bchannels
        for ch in range(0,  self.top_channels):
            s2o = int(( ch % self.neighborhood_grid_width_ + x_shift ) * self.stride2) + self.pad_shift
#            print('s2o:  ', s2o)
            if s2o > 0 :
#                print(img1[:, :, :, :-s2o].shape , img2[:, :, :, s2o:].shape )
                top[:,ch,:, :-s2o] = self.corr_func(img1[:, :, :, :-s2o] , img2[:, :, :, s2o:])
            elif s2o < 0:
#                print(img1[:, :, :, -s2o:].shape, img2[:, :, :, :s2o].shape )
                top[:, ch, :, -s2o:] = self.corr_func(img1[:, :, :, -s2o:], img2[:, :, :, :s2o])
            else:
                top[:, ch, :, :]  = self.corr_func(img1, img2)

        top = top / sumelems
        return top
    def forward(self, l_in, r_in):
        l_g_in = torch.split(l_in, self.num_group, dim=1)
        r_g_in = torch.split(r_in, self.num_group, dim=1)
        corr_list = []
        for i in range(self.num_group):
            corr_tmp = self.corr_forward(l_g_in[i], r_g_in[i])
            corr_list.append(corr_tmp)
        corr_out = torch.cat(corr_list, dim=1) #/l_g_in[0].shape[1]
        # print('corr_out:{}'.format(corr_out.size()))
        return corr_out

#print('begin test...')
#a1 = torch.rand(1,64,192,192)
#a2 = torch.rand(1,64,192,192)
#m = Corr1d_x()
#print('tranditon...')
#b = m(a1, a2)
#print('b...')
#print(b[0,0,:,:])
#
#from boxx import loga
#print('new')
#b1 = m.forward1(a1, a2)
#print("b1...")
#print(b1[0,0,:,:])


class Conv2d(nn.Conv2d):
    def forward(self, obj_in):
        obj_out = super(Conv2d, self).forward(obj_in)
        if(flag_check_shape):
            print(msg_conv(obj_in, self, obj_out))
        return obj_out
        
class ConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, obj_in):
        obj_out = super(ConvTranspose2d, self).forward(obj_in)
        if(flag_check_shape):
            print(msg_conv(obj_in, self, obj_out))
        return obj_out
        
class Conv3d(nn.Conv3d):
    def forward(self, obj_in):
        obj_out =super(Conv3d, self).forward(obj_in)
        if(flag_check_shape):
            print(msg_conv(obj_in, self, obj_out))
        return obj_out
        
class ConvTranspose3d(nn.ConvTranspose3d):
    def forward(self, obj_in):
        obj_out = super(ConvTranspose3d, self).forward(obj_in)
        if(flag_check_shape):
            print(msg_conv(obj_in, self.conv, obj_out))
        return obj_out
        
class CondNorm(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, nhidden=128):
        self.norm_param_free = nn.InstanceNorm2d(out_planes, affine=False)
        self.conv_share = nn.Sequential(
                              nn.Conv2d(inplanes, nhidden, kernel_size, stride),
                              nn.ReLU()
                              )
        self.conv_gamma = nn.Conv2d(nhidden, out_planes, kernel_size, stride)
        self.conv_beta = nn.Conv2d(nhidden, out_planes, kernel_size, stride)

    def forward(self, x, y):
        normed_x = self.norm_param_free(x)

        y = F.interpolate(y, size=x.size()[2:], mode='nearest')
        actv = self.conv_share(y)
        gamma = self.conv_gamma(actv)
        beta = self.conv_beta(actv)
        return normed_x * gamma + beta


def conv2d_bn(in_planes, out_planes, kernel_size=3, stride=1, with_bias=True, bn=flag_bn, activefun=nn.LeakyReLU(negative_slope = 0.1, inplace=True), instance = False):
    "2d convolution with padding, bn and activefun; activefun('PReLU, LeakyReLU')"
    assert kernel_size % 2 == 1
    conv2d = Conv2d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=with_bias)
    
    if(not bn and not activefun): 
        return conv2d
    
    layers = []
    layers.append(conv2d)
    if bn: layers.append(nn.BatchNorm2d(out_planes))
    if instance: layers.append(nn.InstanceNorm2d(out_planes))
    if activefun: layers.append(activefun)
    
    return nn.Sequential(*layers)


def deconv2d_bn(in_planes, out_planes, kernel_size=4, stride=2, with_bias=True, bn=flag_bn, activefun = None):
    "2d deconvolution with padding, bn and activefun"
    assert stride > 1
    p = (kernel_size - 1)//2
    op = stride - (kernel_size - 2*p)
    conv2d = ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding=p, output_padding=op, bias=with_bias)
    
    if(not bn and not activefun): 
        return conv2d
    
    layers = []
    layers.append(conv2d)
    if bn: layers.append(nn.BatchNorm2d(out_planes))
    if activefun: layers.append(activefun)
    
    return nn.Sequential(*layers)


def conv3d_bn(in_planes, out_planes, kernel_size=3, stride=1, with_bias=True, bn=flag_bn, activefun=activefun_default):
    "3d convolution with padding, bn and activefun"
    conv3d = Conv3d(in_planes, out_planes, kernel_size, stride, padding=(kernel_size - 1)//2, bias=with_bias)

    if(not bn and not activefun): 
        return conv3d

    layers = []
    layers.append(conv3d)
    if bn: layers.append(nn.BatchNorm3d(out_planes))
    if activefun: layers.append(activefun)

    return nn.Sequential(*layers)

def deconv3d_bn(in_planes, out_planes, kernel_size=4, stride=2, with_bias=True, bn=flag_bn, activefun=activefun_default):
    "3d deconvolution with padding, bn and activefun"
    assert stride > 1
    p = (kernel_size - 1)//2
    op = stride - (kernel_size - 2*p)
    conv2d = ConvTranspose3d(in_planes, out_planes, kernel_size, stride, padding=p, output_padding=op, bias=with_bias)
    
    if(not bn and not activefun): 
        return conv2d
    
    layers = []
    layers.append(conv2d)
    if bn: layers.append(nn.BatchNorm2d(out_planes))
    if activefun: layers.append(activefun)
    
    return nn.Sequential(*layers)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def make_layer_res(block, blocks, inplanes, planes, stride=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)

def conv_res(inplanes, planes, blocks, stride=1):
    block = BasicBlock
    return make_layer_res(block, blocks, inplanes, planes, stride)

def conv_res_bottleneck(inplanes, planes, blocks, stride=1):
    block = Bottleneck
    return make_layer_res(block, blocks, inplanes, planes, stride)


def save_checkpoint(state_dict, path):
    torch.save(state_dict, path)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


class Octave_conv(nn.Module):
    def __init__(self, H_in_c, L_in_c, H_out_c, L_out_c, kernel_size, stride):
        super(Octave_conv, self).__init__()
        self.W_hh = conv2d_bn(H_in_c, H_out_c, kernel_size, stride)
        self.W_hl = conv2d_bn(H_in_c, L_out_c, kernel_size, stride)
        self.W_ll = conv2d_bn(L_in_c, L_out_c, kernel_size, stride)
        self.W_lh = conv2d_bn(L_in_c, H_out_c, kernel_size, stride)
        self.PRelu = activefun_default

    def forward(self, X_h, X_l):

        Y_hh = self.PRelu(self.W_hh(X_h))
        Y_hl = self.PRelu(self.W_hl(F.avg_pool2d(X_h, kernel_size = 2)))
        Y_lh = F.interpolate(self.PRelu(self.W_lh(X_l)), scale_factor= 2, mode='nearest')
        Y_ll = self.PRelu(self.W_ll(X_l))
        # print('hh',Y_hh.shape,'hl', Y_hl.shape,'lh', Y_lh.shape,'ll', Y_ll.shape)
        Y_h = Y_hh + Y_lh
        Y_l = Y_ll + Y_hl
        return Y_h, Y_l


def load_multi_gpu_checkpoint(net, pkl_path, name):
    print(pkl_path)
    pretrained_dict = torch.load(pkl_path)[name]
    net = nn.DataParallel(net)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net = net.module
    #print('load the checkpoint from {}'.format(pkl_path))
    #state = torch.load(pkl_path)
    #net = nn.DataParallel(net)
    #net.load_state_dict(state['model_state_dict'])
    #net = net.module
    #print('val loss: {}'.format(state['val_loss']))
    return net

def load_checkpoint(net, pkl_path, dst_device):
    state = torch.load(pkl_path, map_location=dst_device)
    net.load_state_dict(state['model_state_dict'])
    print('val loss: {}'.format(state['val_loss']))
    return net

def load_part_checkpoint(net, pkl_path, dst_device):
    state = torch.load(pkl_path)
    pretrained_dict = state#['model_state_dict']
    model_dict = net.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    no_pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in model_dict}
    print(no_pretrained_dict)
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    net.load_state_dict(model_dict)
    # net.load_state_dict(state['model_state_dict'])
    # print('val loss: {}'.format(state['val_loss']))
    if not pretrained_dict:
        print("=> empty overlap between model and pretrained_model!!!")
    else:
        print("=> loaded checkpoint '{}'".format(pkl_path))
    return net

def hyperparameters2string(args):
    pass

def disparity_regression(x, maxdisp, pad_shift, level):
    assert len(x.shape) == 4
    disp_values = torch.arange(level * (-maxdisp - pad_shift), level*(maxdisp - pad_shift+1), dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, level*(2*maxdisp + 1) , 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)

def disp_reg(x, left_disp, right_disp):
    disp_values = torch.arange(left_disp, right_disp, dtype=x.dtype, device=x.device)
    

def weight_init_(m):
    for m in m.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv3d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()


