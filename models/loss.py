import torch.nn.functional as F


def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def model_loss0(disp_ests, disp_gt, mask):
    scale = [0, 1, 2, 3, 4, 5, 6]
    weights = [1, 1, 1, 0.8, 0.6, 0.4, 0.2]
    all_losses = []
    for disp_est, weight, s in zip(disp_ests, weights, scale):
        if s != 0:
            dgt = F.interpolate(disp_gt, scale_factor=1/(2**s))
            m = F.interpolate(mask.float(), scale_factor=1/(2**s)).byte()
        else:
            dgt = disp_gt
            m = mask
        all_losses.append(weight * F.smooth_l1_loss(disp_est[m], dgt[m], size_average=True))
    return sum(all_losses)

def warp_loss(gen, real, weights=[0.5,0.5,0.7]):
    #weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for g0, r, weight in zip(gen, real, weights):
        g, m = g0
        m = m.float()
        #perm = torch.randperm(g.size(1))[:3]
        #m = (g[:,perm,:,:].abs() > 1e-4)
        #all_losses.append(weight * F.l1_loss(g[:,perm,:,:][m], r[:,perm,:,:][m], size_average=True))
        all_losses.append(weight * (m * F.l1_loss(g, r, reduction='none').mean(1)).mean())
    return sum(all_losses)

