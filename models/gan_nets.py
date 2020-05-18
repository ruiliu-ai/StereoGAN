import torch.nn as nn
import torch.nn.functional as F
import torch
from .bilinear_sampler import bilinear_sampler
from .loss import warp_loss

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 2, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        self.model = nn.Sequential(*model)

        # Upsampling
        out_features //= 2
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        )
        in_features = out_features
        out_features //= 2
        self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
        )
        in_features = out_features

        # Output layer
        #model += [nn.ReflectionPad2d((1,1,1,0)), nn.Conv2d(out_features, channels, 3, stride=1), nn.Tanh()]
        out_layer = [nn.ReflectionPad2d(3), nn.Conv2d(out_features, channels, 7, stride=1), nn.Tanh()]

        self.out_layer = nn.Sequential(*out_layer)

    def forward(self, x, offset=None, extract_feat=False, feat_gt=None, zx=False, zx_relax=None):
        num, _, height, width = x.size()
        x = self.model(x)
        x1 = self.up1(x)
        x2 = self.up2(x1)
        if zx: 
            z = torch.randn(num, 1, height//4, width//4).cuda() * 0.01
            k = torch.cuda.LongTensor(1).random_(0, x.size(1))
            x_a = x.clone()
            x_a[:, [k], :, :] = x_a[:, [k], :, :] + z
            x1_a = self.up1(x_a)
            x2_a = self.up2(x1_a)
            out = self.out_layer(x2)
            out_a = self.out_layer(x2_a)
            lz = torch.mean(torch.abs(out-out_a)) / torch.mean(torch.abs(z))
            if zx_relax:
                loss_lz = torch.abs(1/(lz+1e-5) - 0)
            elif zx_relax is False:
                loss_lz = lz
            else:
                raise "Non-supportive relax"
            return loss_lz
        if offset is not None:
            if extract_feat:
                if len(offset) == 3:
                    y = bilinear_sampler(x, offset[-1], 'zeros')
                    y1 = bilinear_sampler(x1, offset[-2], 'zeros')
                    y2 = bilinear_sampler(x2, offset[-3], 'zeros')
                    loss_warp = warp_loss([y, y1, y2], feat_gt, weights=[0.5,0.5,0.7])
                    return loss_warp
                else:
                    y = bilinear_sampler(x, F.max_pool2d(offset,4,4)/4, 'zeros')
                    y1 = bilinear_sampler(x1, F.max_pool2d(offset,2,2)/2, 'zeros')
                    y2 = bilinear_sampler(x2, offset, 'zeros')
                if feat_gt is not None:
                    loss_warp = warp_loss([y, y1, y2], feat_gt, weights=[0.5,0.5,0.7])
                    return bilinear_sampler(self.out_layer(x2), offset, 'zeros'), loss_warp
                return bilinear_sampler(self.out_layer(x2), offset, 'zeros'), [y, y1, y2]
            return bilinear_sampler(self.out_layer(x2), offset, 'zeros')
        if extract_feat:
            return self.out_layer(x2), [x, x1, x2]
        return self.out_layer(x2)

    def forward_R(self, x):
        num, _, height, width = x.size()
        x = self.model(x)
        z = torch.randn(num, 1, height//4, width//4).cuda() * 0.01
        k = torch.cuda.LongTensor(1).random_(0, x.size(1))
        x[:, [k], :, :] = x[:, [k], :, :] + z
        x1 = self.up1(x)
        x2 = self.up2(x1)
        return self.out_layer(x2)

class GeneratorResUNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResUNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features

        # Downsampling
        out_features *= 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features
        out_features *= 2
        self.enc3 = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features

        # Residual blocks
        resblk = []
        for _ in range(num_residual_blocks):
            resblk += [ResidualBlock(out_features)]
        self.resblk = nn.Sequential(*resblk)

        # Upsampling
        out_features //= 2
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features
        out_features //= 2
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )
        in_features = out_features

        # Output layer
        self.dec1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(out_features, channels, 7, stride=1),
            nn.Tanh(),
        )

    def forward(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        out = self.resblk(h3) + h3
        out = self.dec3(out) + h2
        out = self.dec2(out) + h1
        out = self.dec1(out)
        return out


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        self.conv.weight = torch.nn.Parameter(torch.Tensor([[[[1,0,-1],[2,0,-2],[1,0,-1]]]]))
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.conv(x)

#sobel = Sobel()
#print(sobel.conv.weight)

##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        model = []
        model += discriminator_block(in_channels, 64, normalization=False)
        model += discriminator_block(64, 128)
        model += discriminator_block(128, 256)
        model += discriminator_block(256, 512)
        model += [nn.ZeroPad2d((1, 0, 1, 0))]
        model += [nn.Conv2d(512, 1, 4, padding=1, bias=False)]

        self.model = nn.Sequential(*model)

    def forward(self, img_input):
        # Concatenate image and condition image by channels to produce input
        #img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
