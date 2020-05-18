import time
import os
import argparse
import sys
import itertools
import numpy as np
from scipy import misc

import torch
import torch.nn as nn 
import torch.optim as  optim
import torch.nn.functional as F
from models.loss import warp_loss, model_loss0
from models.dispnet import dispnetcorr
from models.gan_nets import GeneratorResNet, Discriminator, weights_init_normal
from dataset import ImageDataset, ValJointImageDataset

from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from utils.util import AverageMeter
from utils.util import load_multi_gpu_checkpoint, load_checkpoint
from utils.metric_utils.metrics import *
from utils import pytorch_ssim

def val(valloader, net, writer, epoch=1, board_save=True):
    net.eval()
    EPEs, D1s, Thres1s, Thres2s, Thres3s = 0, 0, 0, 0, 0
    i = 0
    for left_img, right_img, disp_gt in valloader:
        left_img = left_img.cuda()
        right_img = right_img.cuda()
        disp_gt = disp_gt.cuda()
        i = i + 1
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
        disp_est = net(left_img, right_img)[0].squeeze(1)
        EPEs += EPE_metric(disp_est, disp_gt, mask)
        D1s += D1_metric(disp_est, disp_gt, mask)
        Thres1s += Thres_metric(disp_est, disp_gt, mask, 2.0)
        Thres2s += Thres_metric(disp_est, disp_gt, mask, 4.0)
        Thres3s += Thres_metric(disp_est, disp_gt, mask, 5.0)
    if board_save:
        writer.add_scalar("val/EPE", EPEs/i, epoch)
        writer.add_scalar("val/D1", D1s/i, epoch)
        writer.add_scalar("val/Thres2", Thres1s/i, epoch)
        writer.add_scalar("val/Thres4", Thres2s/i, epoch)
        writer.add_scalar("val/Thres5", Thres3s/i, epoch)
    return EPEs/i, D1s/i

def train(args):
    writer = SummaryWriter(comment=args.writer)
    os.makedirs(args.checkpoint_save_path, exist_ok=True)

    argsDict = args.__dict__
    for k,v in argsDict.items():
        writer.add_text('hyperparameter', '{} : {}'.format(str(k), str(v)))

    print_freq = args.print_freq
    test_freq = 1
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    
    input_shape = (3, args.img_height, args.img_width)
    net = dispnetcorr(args.maxdisp)
    G_AB = GeneratorResNet(input_shape, 2)
    G_BA = GeneratorResNet(input_shape, 2)
    D_A = Discriminator(3)
    D_B = Discriminator(3)

    if args.load_checkpoints:
        if args.load_from_mgpus_model:
            if args.load_dispnet_path:
                net = load_multi_gpu_checkpoint(net, args.load_dispnet_path, 'model')
            else:
                net.apply(weights_init_normal)
            G_AB = load_multi_gpu_checkpoint(G_AB, args.load_gan_path, 'G_AB')
            G_BA = load_multi_gpu_checkpoint(G_BA, args.load_gan_path, 'G_BA')
            D_A = load_multi_gpu_checkpoint(D_A, args.load_gan_path, 'D_A')
            D_B = load_multi_gpu_checkpoint(D_B, args.load_gan_path, 'D_B')
        else:
            if args.load_dispnet_path:
                net = load_checkpoint(net, args.load_checkpoint_path, device)
            else:
                net.apply(weights_init_normal)
            G_AB = load_checkpoint(G_AB, args.load_gan_path, 'G_AB')
            G_BA = load_checkpoint(G_BA, args.load_gan_path, 'G_BA')
            D_A = load_checkpoint(D_A, args.load_gan_path, 'D_A')
            D_B = load_checkpoint(D_B, args.load_gan_path, 'D_B')
    else:
        net.apply(weights_init_normal)
        G_AB.apply(weights_init_normal)
        G_BA.apply(weights_init_normal)
        D_A.apply(weights_init_normal)
        D_B.apply(weights_init_normal)

    # optimizer = optim.SGD(params, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=args.lr_rate, betas=(0.9, 0.999))
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr_gan, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=args.lr_gan, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=args.lr_gan, betas=(0.5, 0.999))

    if args.use_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net, device_ids=list(range(args.use_multi_gpu)))
        G_AB = nn.DataParallel(G_AB, device_ids=list(range(args.use_multi_gpu)))
        G_BA = nn.DataParallel(G_BA, device_ids=list(range(args.use_multi_gpu)))
        D_A = nn.DataParallel(D_A, device_ids=list(range(args.use_multi_gpu)))
        D_B = nn.DataParallel(D_B, device_ids=list(range(args.use_multi_gpu)))

    net.to(device)
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)

    criterion_GAN = torch.nn.MSELoss().cuda()
    criterion_identity = torch.nn.L1Loss().cuda()
    ssim_loss = pytorch_ssim.SSIM()

    # data loader
    if args.source_dataset == 'driving':
        dataset = ImageDataset(height=args.img_height, width=args.img_width)
    elif args.source_dataset == 'synthia':
        dataset = ImageDataset2(height=args.img_height, width=args.img_width)
    else:
        raise "No suportive dataset"
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valdataset = ValJointImageDataset()
    valloader = torch.utils.data.DataLoader(valdataset, batch_size=args.test_batch_size, shuffle=False, num_workers=1)

    train_loss_meter = AverageMeter()
    val_loss_meter = AverageMeter()

    ## debug only
    #with torch.no_grad():
    #    l1_test_loss, out_val = val(valloader, net, G_AB, None, writer, epoch=0, board_save=True)
    #    val_loss_meter.update(l1_test_loss)
    #    print('Val epoch[{}/{}] loss: {}'.format(0, args.total_epochs, l1_test_loss))

    print('begin training...')
    best_val_d1 = 1.
    best_val_epe = 100.
    for epoch in range(args.total_epochs):
        #net.train()
        #G_AB.train()

        n_iter = 0
        running_loss = 0.
        t = time.time()
        # custom lr decay, or warm-up
        lr = args.lr_rate
        if epoch >= int(args.lrepochs.split(':')[0]):
            lr = lr / int(args.lrepochs.split(':')[1])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, batch in enumerate(trainloader):
            n_iter += 1
            leftA = batch['leftA'].to(device)
            rightA = batch['rightA'].to(device)
            leftB = batch['leftB'].to(device)
            rightB = batch['rightB'].to(device)
            dispA = batch['dispA'].unsqueeze(1).float().to(device)
            dispB = batch['dispB'].to(device) 
            out_shape = (leftA.size(0), 1, args.img_height//16, args.img_width//16)
            valid = torch.cuda.FloatTensor(np.ones(out_shape))
            fake = torch.cuda.FloatTensor(np.zeros(out_shape))
            
            if i % args.train_ratio_gan == 0:
                # train generators
                G_AB.train()
                G_BA.train()
                net.eval()
                optimizer_G.zero_grad()

                # Identity loss
                loss_id_A = (criterion_identity(G_BA(leftA), leftA) + criterion_identity(G_BA(rightA), rightA)) / 2
                loss_id_B = (criterion_identity(G_AB(leftB), leftB) + criterion_identity(G_AB(rightB), rightB)) / 2
                loss_id = (loss_id_A + loss_id_B) / 2

                if args.lambda_warp_inv:
                    fake_leftB, fake_leftB_feats = G_AB(leftA, extract_feat=True)
                    fake_leftA, fake_leftA_feats = G_BA(leftB, extract_feat=True)
                else:
                    fake_leftB = G_AB(leftA)
                    fake_leftA = G_BA(leftB)
                if args.lambda_warp:
                    fake_rightB, fake_rightB_feats = G_AB(rightA, extract_feat=True)
                    fake_rightA, fake_rightA_feats = G_BA(rightB, extract_feat=True)
                else:
                    fake_rightB = G_AB(rightA)
                    fake_rightA = G_BA(rightB)
                loss_GAN_AB = criterion_GAN(D_B(fake_leftB), valid)
                loss_GAN_BA = criterion_GAN(D_A(fake_leftA), valid)
                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                if args.lambda_warp_inv:
                    rec_leftA, rec_leftA_feats = G_BA(fake_leftB, extract_feat=True)
                else:
                    rec_leftA = G_BA(fake_leftB)
                if args.lambda_warp:
                    rec_rightA, rec_rightA_feats = G_BA(fake_rightB, extract_feat=True)
                else:
                    rec_rightA = G_BA(fake_rightB)
                rec_leftB = G_AB(fake_leftA)
                rec_rightB = G_AB(fake_rightA)
                loss_cycle_A = (criterion_identity(rec_leftA, leftA) + criterion_identity(rec_rightA, rightA)) / 2
                loss_ssim_A = 1. - (ssim_loss(rec_leftA, leftA) + ssim_loss(rec_rightA, rightA)) / 2
                loss_cycle_B = (criterion_identity(rec_leftB, leftB) + criterion_identity(rec_rightB, rightB)) / 2
                loss_ssim_B = 1. - (ssim_loss(rec_leftB, leftB) + ssim_loss(rec_rightB, rightB)) / 2
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
                loss_ssim = (loss_ssim_A + loss_ssim_B) / 2

                # mode seeking loss
                if args.lambda_ms:
                    loss_ms = G_AB(leftA, zx=True, zx_relax=True).mean()
                else:
                    loss_ms = 0

                # warping loss
                if args.lambda_warp_inv:
                    fake_leftB_warp, loss_warp_inv_feat1 = G_AB(rightA, -dispA, True, [x.detach() for x in fake_leftB_feats])
                    rec_leftA_warp, loss_warp_inv_feat2 = G_BA(fake_rightB, -dispA, True, [x.detach() for x in rec_leftA_feats])
                    loss_warp_inv1 = warp_loss([(G_BA(fake_leftB_warp[0]), fake_leftB_warp[1])], [leftA], weights=[1])
                    loss_warp_inv2 = warp_loss([rec_leftA_warp], [leftA], weights=[1])
                    loss_warp_inv = loss_warp_inv1 + loss_warp_inv2 + loss_warp_inv_feat1.mean() + loss_warp_inv_feat2.mean()
                else:
                    loss_warp_inv = 0

                if args.lambda_warp:
                    fake_rightB_warp, loss_warp_feat1 = G_AB(leftA, dispA, True, [x.detach() for x in fake_rightB_feats])
                    rec_rightA_warp, loss_warp_feat2 = G_BA(fake_leftB, dispA, True, [x.detach() for x in rec_rightA_feats])
                    loss_warp1 = warp_loss([(G_BA(fake_rightB_warp[0]), fake_rightB_warp[1])], [rightA], weights=[1])
                    loss_warp2 = warp_loss([rec_rightA_warp], [rightA], weights=[1])
                    loss_warp = loss_warp1 + loss_warp2 + loss_warp_feat1.mean() + loss_warp_feat2.mean()
                else:
                    loss_warp = 0

                # corr loss
                if args.lambda_corr:
                    corrB = net(leftB, rightB, extract_feat=True)
                    corrB1 = net(leftB, rec_rightB, extract_feat=True)
                    corrB2 = net(rec_leftB, rightB, extract_feat=True)
                    corrB3 = net(rec_leftB, rec_rightB, extract_feat=True)
                    loss_corr = (criterion_identity(corrB1, corrB)+criterion_identity(corrB2, corrB)+criterion_identity(corrB3, corrB))/3
                else:
                    loss_corr = 0.

                lambda_ms = args.lambda_ms * (args.total_epochs - epoch) / args.total_epochs
                loss_G = loss_GAN + args.lambda_cycle*(args.alpha_ssim*loss_ssim+(1-args.alpha_ssim)*loss_cycle) + args.lambda_id*loss_id \
                       + args.lambda_warp*loss_warp + args.lambda_warp_inv*loss_warp_inv + args.lambda_corr*loss_corr + lambda_ms*loss_ms 
                loss_G.backward()
                optimizer_G.step()

                # train discriminators. A: real, B: syn
                optimizer_D_A.zero_grad()
                loss_real_A = criterion_GAN(D_A(leftA), valid)
                fake_leftA.detach_()
                loss_fake_A = criterion_GAN(D_A(fake_leftA), fake)
                loss_D_A = (loss_real_A + loss_fake_A) / 2
                loss_D_A.backward()
                optimizer_D_A.step()
                
                optimizer_D_B.zero_grad()
                #loss_real_B = criterion_GAN(D_B(torch.cat([syn_left_img, syn_right_img], 0)), valid)
                #fake_syn_left.detach_()
                #fake_syn_right.detach_()
                #loss_fake_B = criterion_GAN(D_B(torch.cat([fake_syn_left, fake_syn_right], 0)), fake)
                loss_real_B = criterion_GAN(D_B(leftB), valid)
                fake_leftB.detach_()
                loss_fake_B = criterion_GAN(D_B(fake_leftB), fake)
                loss_D_B = (loss_real_B + loss_fake_B) / 2
                loss_D_B.backward()
                optimizer_D_B.step()

            # train disp net
            net.train()
            G_AB.eval()
            G_BA.eval()
            optimizer.zero_grad()
            disp_ests = net(G_AB(leftA), G_AB.forward(rightA))
            mask = (dispA < args.maxdisp) & (dispA > 0)
            loss0 = model_loss0(disp_ests, dispA, mask)

            if args.lambda_disp_warp_inv:
                disp_warp = [-disp_ests[i] for i in range(3)]
                loss_disp_warp_inv = G_BA(rightB, disp_warp, True, [x.detach() for x in fake_leftA_feats])
                loss_disp_warp_inv = loss_disp_warp_inv.mean()
            else:
                loss_disp_warp_inv = 0

            if args.lambda_disp_warp:
                disp_warp = [disp_ests[i] for i in range(3)]
                loss_disp_warp = G_BA(leftB, disp_warp, True, [x.detach() for x in fake_rightA_feats])
                loss_disp_warp = loss_disp_warp.mean()
            else:
                loss_disp_warp = 0

            loss = loss0 + args.lambda_disp_warp*loss_disp_warp + args.lambda_disp_warp_inv*loss_disp_warp_inv
            loss.backward()
            optimizer.step()

            if i % print_freq == print_freq - 1:
                print('epoch[{}/{}]  step[{}/{}]  loss: {}'.format(epoch, args.total_epochs, i, len(trainloader), loss.item() ))
                train_loss_meter.update(running_loss / print_freq)
                #writer.add_scalar('loss/trainloss avg_meter', train_loss_meter.val, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_disp', loss0, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_disp_warp', loss_disp_warp, train_loss_meter.count * print_freq)
                writer.add_scalar('loss/loss_disp_warp_inv', loss_disp_warp_inv, train_loss_meter.count * print_freq)

                if i % args.train_ratio_gan == 0:
                    writer.add_scalar('loss/loss_G', loss_G, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_gan', loss_GAN, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_cycle', loss_cycle, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_id', loss_id, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_warp', loss_warp, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_warp_inv', loss_warp_inv, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_corr', loss_corr, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_ms', loss_ms, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_D_A', loss_D_A, train_loss_meter.count * print_freq)
                    writer.add_scalar('loss/loss_D_B', loss_D_B, train_loss_meter.count * print_freq)

                    imgA_visual = vutils.make_grid(leftA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    fakeB_visual = vutils.make_grid(fake_leftB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    recA_visual = vutils.make_grid(rec_leftA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    rightA_visual = vutils.make_grid(rightA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    fakeB_R_visual = vutils.make_grid(fake_rightB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    recA_R_visual = vutils.make_grid(rec_rightA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)

                    imgB_visual = vutils.make_grid(leftB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    fakeA_visual = vutils.make_grid(fake_leftA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    recB_visual = vutils.make_grid(rec_leftB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    rightB_visual = vutils.make_grid(rightB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    fakeA_R_visual = vutils.make_grid(fake_rightA[:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    recB_R_visual = vutils.make_grid(rec_rightB[:4,:,:,:], nrow=1, normalize=True, scale_each=True)

                    writer.add_image('ABA_L/imgA', imgA_visual, i)
                    writer.add_image('ABA_L/fakeB', fakeB_visual, i)
                    writer.add_image('ABA_L/recA', recA_visual, i)
                    writer.add_image('ABA_R/imgA', rightA_visual, i)
                    writer.add_image('ABA_R/fakeB', fakeB_R_visual, i)
                    writer.add_image('ABA_R/recA', recA_R_visual, i)
                    writer.add_image('BAB_L/imgB', imgB_visual, i)
                    writer.add_image('BAB_L/fakeA', fakeA_visual, i)
                    writer.add_image('BAB_L/recB', recB_visual, i)
                    writer.add_image('BAB_R/imgB', rightB_visual, i)
                    writer.add_image('BAB_R/fakeA', fakeA_R_visual, i)
                    writer.add_image('BAB_R/recB', recB_R_visual, i)

                if args.lambda_warp_inv:
                    recA_warp_visual = vutils.make_grid(rec_leftA_warp[0][:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    fakeB_warp_visual = vutils.make_grid(fake_leftB_warp[0][:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    writer.add_image('warp/recA_L_warp', recA_warp_visual, i)
                    writer.add_image('warp/fakeB_L_warp', fakeB_warp_visual, i)
                if args.lambda_warp:
                    writer.add_image('warp/recA_R_warp', recA_warp_R_visual, i)
                    writer.add_image('warp/fakeB_R_warp', fakeB_warp_R_visual, i)
                    recA_warp_R_visual = vutils.make_grid(rec_rightA_warp[0][:4,:,:,:], nrow=1, normalize=True, scale_each=True)
                    fakeB_warp_R_visual = vutils.make_grid(fake_rightB_warp[0][:4,:,:,:], nrow=1, normalize=True, scale_each=True)

        with torch.no_grad():
            EPE, D1 = val(valloader, net, writer, epoch=epoch, board_save=True)

        t1 = time.time()
        print('epoch:{}, D1:{:.6f}, EPE:{:.6f}, cost time:{} '.format(epoch, D1, EPE, t1-t))

        if (epoch % args.save_interval == 0) or D1 < best_val_d1 or EPE < best_val_epe:
            best_val_d1 = D1
            best_val_epe = EPE
            torch.save({
                        'epoch': epoch,
                        'G_AB': G_AB.state_dict(),
                        'G_BA': G_BA.state_dict(),
                        'D_A': D_A.state_dict(),
                        'D_B': D_B.state_dict(),
                        'model': net.state_dict(),
                        'optimizer_DA_state_dict': optimizer_D_A.state_dict(),
                        'optimizer_DB_state_dict': optimizer_D_B.state_dict(),
                        'optimizer_G_state_dict': optimizer_G.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, args.checkpoint_save_path + '/ep' + str(epoch) + '_D1_{:.4f}_EPE{:.4f}'.format(D1, EPE) + '.pth.rar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    # dataset
    parser.add_argument('--source_dataset', type=str, default='driving')
    parser.add_argument('--img_height', type=int, default=320)
    parser.add_argument('--img_width', type=int, default=512)

    # training
    parser.add_argument('--lr_rate', nargs='?', type=float, default=1e-4, help='learning rate for dispnetc')
    parser.add_argument('--lrepochs', type=str, default='30:1', help='the epochs to decay lr: the downscale rate')
    parser.add_argument('--lr_gan', nargs='?', type=float, default=2e-4, help='learning rate for GAN')
    parser.add_argument('--train_ratio_gan', nargs='?', type=int, default=5, help='training ratio disp:gan=5:1')
    parser.add_argument('--batch_size', nargs='?', type=int, default=6, help='batch size')
    parser.add_argument('--test_batch_size', nargs='?', type=int, default=4, help='test batch size')
    parser.add_argument('--total_epochs', nargs='?', type=int, default='201')
    parser.add_argument('--save_interval', nargs='?', type=int, default='10')
    parser.add_argument('--model_type', nargs='?', type=str, default='dispnetc')
    parser.add_argument('--maxdisp', type=int, default=192)

    # hyper params
    parser.add_argument('--lambda_cycle', type=float, default=10)
    parser.add_argument('--alpha_ssim', type=float, default=0.85)
    parser.add_argument('--lambda_id', type=float, default=5)
    parser.add_argument('--lambda_ms', type=float, default=1)
    parser.add_argument('--lambda_warp', type=float, default=0)
    parser.add_argument('--lambda_warp_inv', type=float, default=1)
    parser.add_argument('--lambda_disp_warp', type=float, default=0)
    parser.add_argument('--lambda_disp_warp_inv', type=float, default=1)
    parser.add_argument('--lambda_corr', type=float, default=10)

    # load & save checkpoints
    parser.add_argument('--load_checkpoints', nargs='?', type=int, default=0, help='load from ckp(saved by Pytorch)')
    parser.add_argument('--load_from_mgpus_model', nargs='?', type=int, default=0, help='load ckp which is saved by mgus(nn.DataParallel)')
    parser.add_argument('--load_gan_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    parser.add_argument('--load_dispnet_path', nargs='?', type=str, default=None, help='path of ckp(saved by Pytorch)')
    parser.add_argument('--checkpoint_save_path', nargs='?', type=str, default='checkpoints/best_checkpoint.pth.tar')

    # tensorboard, print freq
    parser.add_argument('--writer', nargs='?', type=str, default='StereoGAN')
    parser.add_argument('--print_freq', '-p', default=150, type=int, metavar='N', help='print frequency (default: 150)')

    # other
    parser.add_argument('--use_multi_gpu', nargs='?', type=int, default=0, help='the number of multi gpu to use')
    args = parser.parse_args()
    train(args)
