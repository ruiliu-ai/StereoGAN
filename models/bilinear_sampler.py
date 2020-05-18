import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
    return torch.index_select(a, dim, order_index)

def _interpolate(im, x, y, warp_mode):
    bs, c, h, w = im.size()
    edge_size = 0
    if warp_mode == 'border':
        edge_size = 1
        im = nn.ZeroPad2d(1)(im)
        x = x + edge_size
        y = y + edge_size
    elif warp_mode == 'edge':
        edge_size = 0
    else:
        raise 'NotImplementError'

    x = x.clamp(0., w - 1 + 2*edge_size)
    x0_f = torch.floor(x)
    y0_f = torch.floor(y)
    x1_f = x0_f + 1

    x0 = x0_f.long()
    y0 = y0_f.long()
    x1 = x1_f.clamp(0., w - 1 + 2*edge_size).long()

    dim2 = torch.cuda.LongTensor([w + 2*edge_size])
    dim1 = dim2 * torch.cuda.LongTensor([h + 2*edge_size])
    base = (torch.arange(bs).cuda() * dim1).view(-1, 1)
    base = _tile(base, 1, w*h).view(-1)
    base_y0 = base + y0 * dim2
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    im_flat = im.permute(0,2,3,1).contiguous().view(-1, c)
    pix_l = im_flat.index_select(0, idx_l)
    pix_r = im_flat.index_select(0, idx_r)

    weight_l = (x1_f - x).unsqueeze(1)
    weight_r = (x - x0_f).unsqueeze(1)
    return weight_l * pix_l + weight_r *pix_r

def bilinear_sampler0(image, offset, warp_mode='border'):
    bs, c, h, w = image.size()

    x_t, y_t = torch.meshgrid([torch.arange(w), torch.arange(h)])
    x_t_flat = x_t.transpose(1,0).float().contiguous().view(1, -1).cuda()
    y_t_flat = y_t.transpose(1,0).float().contiguous().view(1, -1).cuda()
    #print(x_t_flat)
    #print(y_t_flat)
    x_t_flat = _tile(x_t_flat, 0, bs).view(-1)
    y_t_flat = _tile(y_t_flat, 0, bs).view(-1)

    x_t_flat = x_t_flat + offset.view(-1)
    image = _interpolate(image, x_t_flat, y_t_flat, warp_mode)
    #print(image)
    image = image.view(bs, h, w, c)
    image = image.permute(0,3,1,2).contiguous()
    return image

def bilinear_sampler(image, offset, mode='border'):
    bs, c, h, w = image.size()
    x, y = torch.meshgrid([torch.arange(w), torch.arange(h)])
    x = x.float().cuda()
    y = y.float().cuda()
    x = _tile(x.transpose(1,0).unsqueeze(0), 0, bs)
    min_x = x.min()
    y = _tile(y.transpose(1,0).unsqueeze(0), 0, bs)
    x = x + offset.squeeze(1)
    mask = (x >= min_x) & (x <= w-1)
    norm_x = ((x/(w-1)) * 2 - 1).unsqueeze(-1)
    norm_y = ((y/(h-1)) * 2 - 1).unsqueeze(-1)
    grid = torch.cat([norm_x, norm_y], -1)
    #new_imgs = []
    #for i in range(bs):
    #    new_imgs.append(F.grid_sample(image[[i]], grid[[i]], padding_mode=mode))
    #return torch.cat([new_img, 0])
    return (F.grid_sample(image, grid, padding_mode=mode), mask.squeeze(1))

class BilinearSampler(nn.Module):
    def __init__(self):
        super(BilinearSampler, self).__init__()

    def forward(self, x, offset, mode='border'):
        return bilinear_sampler(x, offset, mode=mode)

#x = torch.arange(24).reshape(1,1,4,6).float().cuda()
#d = torch.ones(1,1,4,6).cuda()
#new = bilinear_sampler(x, d)
#print(x)
#print(new)
#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.w = torch.nn.Parameter(torch.FloatTensor(1))
#        self.w.data.fill_(1)
#
#    def forward(self, x):
#        return x * self.w
#
#x = torch.arange(24).reshape(1,4,6,1)
##print(x)
#x = x.permute(0,3,1,2).float().cuda()
#d = torch.ones(1,4,6).cuda()
#net = Net().cuda()
#
##w = torch.cuda.FloatTensor([1]).requires_grad_(True)
#
#optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
##d = torch.cuda.FloatTensor(1,4,4).uniform_(0,1)
#for i in range(50):
#    y = bilinear_sampler(x, net(d))
#    loss = nn.L1Loss()(y, x)
#    loss.backward()
#    optimizer.step()
#    print(loss)
#    print(net.w.data)
#
##print(y)
