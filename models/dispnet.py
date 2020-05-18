import torch
import torch.nn as nn
from models.util_conv import net_init, conv2d_bn, deconv2d_bn, Corr1d

flag_bias_t = True
flag_bn = False
activefun_t = nn.ReLU(inplace=True)

class dispnetcorr(nn.Module):
    def __init__(self, maxdisparity=192):
        super(dispnetcorr, self).__init__()
        self.name = "dispnetcorr"
        self.D = maxdisparity
        self.delt = 1e-6
        self.count_levels = 7 # 分辨率层数

        # 上采样（2倍）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        # 卷积层
        self.conv1 = conv2d_bn(3, 64, kernel_size=7, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv2 = conv2d_bn(64, 128, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.corr = Corr1d(kernel_size=1, stride=1, D=41, simfun=None)
        self.redir = conv2d_bn(128, 64, kernel_size=1, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)        
        self.conv3a = conv2d_bn(64 + 41, 256, kernel_size=5, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv3b = conv2d_bn(256, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4a = conv2d_bn(256, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv4b = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv5a = conv2d_bn(512, 512, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv5b = conv2d_bn(512, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv6a = conv2d_bn(512, 1024, kernel_size=3, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.conv6b = conv2d_bn(1024, 1024, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        
        # 解卷积层和视差预测层
        self.pr6 = nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv5 = deconv2d_bn(1024, 512, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv5 = conv2d_bn(1025, 512, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr5 = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv4 = deconv2d_bn(512, 256, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv4 = conv2d_bn(769, 256, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv3 = deconv2d_bn(256, 128, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv3 = conv2d_bn(385, 128, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv2 = deconv2d_bn(128, 64, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv2 = conv2d_bn(193, 64, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        self.deconv1 = deconv2d_bn(64, 32, kernel_size=4, stride=2, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.iconv1 = conv2d_bn(97, 32, kernel_size=3, stride=1, flag_bias=flag_bias_t, bn=flag_bn, activefun=activefun_t)
        self.pr1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        
        # 权重初始化
        net_init(self)
        for m in [self.pr6, self.pr5, self.pr4, self.pr3, self.pr2, self.pr1]:
            m.weight.data = m.weight.data*0.1

    def forward(self, imL, imR, extract_feat=False):
        assert imL.shape == imR.shape

        maxD = max(self.D, imL.shape[-1]) # 设置最大视差
        out = []
        out_scale = []

        #　编码阶段
        conv1L = self.conv1(imL)
        conv1R = self.conv1(imR)
        conv2L = self.conv2(conv1L)
        conv2R = self.conv2(conv1R)
        corr = self.corr(conv2L, conv2R)
        if extract_feat:
            return corr
        redir = self.redir(conv2L)
        conv3a = self.conv3a(torch.cat([corr, redir], dim=1))
        conv3b = self.conv3b(conv3a)
        conv4a = self.conv4a(conv3b)
        conv4b = self.conv4b(conv4a)
        conv5a = self.conv5a(conv4b)
        conv5b = self.conv5b(conv5a)
        conv6a = self.conv6a(conv5b)
        conv6b = self.conv6b(conv6a)

        #if extract_feat:
        #    return [conv3b, conv4b, conv5b, conv6b]
        
        #　解码和预测阶段
        pr6 = self.pr6(conv6b)
        #out.insert(0, pr6)
        #out_scale.insert(0, 6)
        pr5 = self.upsample(pr6)
        
        deconv5 = self.deconv5(conv6b)
        iconv5 = self.iconv5(myCat2d(deconv5, pr5, conv5b))
        pr5 = self.pr5(iconv5)
        #out.insert(0, pr5)
        #out_scale.insert(0, 5)
        pr4 = self.upsample(pr5)

        deconv4 = self.deconv4(iconv5)
        iconv4 = self.iconv4(myCat2d(deconv4, pr4, conv4b))
        pr4 = self.pr4(iconv4)
        #out.insert(0, pr4)
        #out_scale.insert(0, 4)
        pr3 = self.upsample(pr4)

        deconv3 = self.deconv3(iconv4)
        iconv3 = self.iconv3(myCat2d(deconv3, pr3, conv3b))
        pr3 = self.pr3(iconv3)
        #out.insert(0, pr3)
        #out_scale.insert(0, 3)
        pr2 = self.upsample(pr3)

        deconv2 = self.deconv2(iconv3)
        iconv2 = self.iconv2(myCat2d(deconv2, pr2, conv2L))
        pr2 = self.pr2(iconv2)
        #out.insert(0, pr2)
        #out_scale.insert(0, 2)
        pr1 = self.upsample(pr2)

        deconv1 = self.deconv1(iconv2)
        iconv1 = self.iconv1(myCat2d(deconv1, pr1, conv1L))
        pr1 = self.pr1(iconv1)
        #out.insert(0, pr1)
        #out_scale.insert(0, 1)
        pr0 = self.upsample(pr1)

        pr = pr0[:, :, :imL.shape[-2], :imL.shape[-1]]
        #out.insert(0, pr0)
        #out_scale.insert(0, 0)

        return [pr,pr1,pr2,pr3,pr4,pr5,pr6]

def myCat2d(*seq):
    assert len(seq[0].shape) == 4
    bn, c, h, w = seq[0].shape
    for tmp in seq:
        _, _, ht, wt = tmp.shape
        if(h > ht): h = ht
        if(w > wt): w = wt
    seq1 = [ seq[i][:, :, :h, :w] for i in range(len(seq))]
    return torch.cat(seq1, dim = 1)

