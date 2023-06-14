import collections

import torch
import torch.nn as nn
import torch.nn.functional as Func
from models.utils import compute_pmp,find_min_pixels,gini
from models.DIAC import DIAC
from models.nn_common import conv, upconv, resnet_block


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        try:
            self.rank = torch.distributed.get_rank()
        except Exception as ex:
            self.rank = 0

        self.device = config.device
        ks = config.ks # kernel size for convolution
        self.Fs = config.Fs # kernel size for IAC
        res_num = config.res_num

        ch1 = config.ch
        ch2 = ch1 * 2
        ch3 = ch1 * 4
        ch4 = ch1 * 4
        self.ch4 = ch4

        # weight init for filter predictor
        self.wiF = config.wiF

        ###################################
        # Feature Extractor - Reconstructor
        ###################################
        # feature extractor
        self.conv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.conv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.conv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.conv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.conv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.conv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.conv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        self.conv4_4 = nn.Sequential(
            conv(2*ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, ch4, kernel_size=ks))

        # reconstructor
        self.conv_res = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=3),
            conv(ch4, ch4, kernel_size=ks))

        self.upconv3_u = upconv(ch4, ch3)
        self.upconv3_1 = resnet_block(ch3, kernel_size=ks, res_num=1)
        self.upconv3_2 = resnet_block(ch3, kernel_size=ks, res_num=1)

        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks, res_num=1)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks, res_num=1)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_1 = resnet_block(ch1, kernel_size=ks, res_num=1)
        self.upconv1_2 = resnet_block(ch1, kernel_size=ks, res_num=1)

        self.out_res = conv(ch1, 3, kernel_size=ks)
        ###################################

        ###################################
        # IFAN
        ###################################
        # filter encoder
        self.kconv1_1 = conv(3, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = conv(ch1, ch1, kernel_size=ks, stride=1)
        self.kconv1_3 = conv(ch1, ch1, kernel_size=ks, stride=1)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = conv(ch2, ch2, kernel_size=ks, stride=1)
        self.kconv2_3 = conv(ch2, ch2, kernel_size=ks, stride=1)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = conv(ch3, ch3, kernel_size=ks, stride=1)
        self.kconv3_3 = conv(ch3, ch3, kernel_size=ks, stride=1)

        self.kconv4_1 = conv(ch3, ch4, kernel_size=ks, stride=2)
        self.kconv4_2 = conv(ch4, ch4, kernel_size=ks, stride=1)
        self.kconv4_3 = conv(ch4, ch4, kernel_size=ks, stride=1)

        # Add the ambiguity prediction layer bpr
        self.bpr = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks, stride=1),
            resnet_block(ch4,kernel_size=ks,res_num=res_num),
            resnet_block(ch4,kernel_size=ks,res_num=res_num),
            conv(ch4, 1, kernel_size=ks, stride=1, act=None)
        )

        # filter predictor
        self.conv_bpr = conv(1, ch4, kernel_size=3)
        self.N = config.N
        self.kernel_dim = self.N * (ch4 * self.Fs * 2) + self.N * ch4
        self.F = nn.Sequential(
            conv(ch4, ch4, kernel_size=ks),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            resnet_block(ch4, kernel_size=ks, res_num=res_num),
            conv(ch4, self.kernel_dim, kernel_size=1, act = None))

    def weights_init_F(self, m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = self.wiF)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def init_F(self):
        self.F.apply(self.weights_init_F)

##########################################################################
    def forward(self, C, GT=None, is_train=False):
        # feature extractor
        f1 = self.conv1_3(self.conv1_2(self.conv1_1(C)))
        f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
        f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
        f_C = self.conv4_3(self.conv4_2(self.conv4_1(f3)))


        # filter encoder
        f = self.kconv1_3(self.kconv1_2(self.kconv1_1(C)))
        f = self.kconv2_3(self.kconv2_2(self.kconv2_1(f)))
        f = self.kconv3_3(self.kconv3_2(self.kconv3_1(f)))
        f = self.kconv4_3(self.kconv4_2(self.kconv4_1(f)))
        # f = self.cbam4(f)

        BPR = self.bpr(f)
        # filter predictor
        f_bpr = self.conv_bpr(BPR)
        f = self.conv4_4(torch.cat([f, f_bpr], dim=1))
        F = self.F(f)
        # IAC
        f = DIAC(f_C, F, self.N, self.ch4, self.Fs)

        # reconstructor
        f = self.conv_res(f)

        f = self.upconv3_u(f) + f3
        f = self.upconv3_2(self.upconv3_1(f))

        f = self.upconv2_u(f) + f2
        f = self.upconv2_2(self.upconv2_1(f))

        f = self.upconv1_u(f) + f1
        f = self.upconv1_2(self.upconv1_1(f))

        out = self.out_res(f) + C

        # results
        outs = collections.OrderedDict()

        if is_train is False:
            outs['result'] = torch.clip(out, 0, 1.0)
        else:
            outs['result'] = out
            # F
            outs['Filter'] = F

            f_e = self.kconv1_3(self.kconv1_2(self.kconv1_1(C)))
            f_e = self.kconv2_3(self.kconv2_2(self.kconv2_1(f_e)))
            f_e = self.kconv3_3(self.kconv3_2(self.kconv3_1(f_e)))
            f_e = self.kconv4_3(self.kconv4_2(self.kconv4_1(f_e)))

            f_G = self.kconv1_3(self.kconv1_2(self.kconv1_1(GT)))
            f_G = self.kconv2_3(self.kconv2_2(self.kconv2_1(f_G)))
            f_G = self.kconv3_3(self.kconv3_2(self.kconv3_1(f_G)))
            f_G = self.kconv4_3(self.kconv4_2(self.kconv4_1(f_G)))
            blur_pred = self.bpr(f_e)
            pmp_c = compute_pmp(f_e)
            pmp_g = compute_pmp(f_G)
            gini_c = gini(pmp_c)
            gini_g = gini(pmp_g)
            blur_esti = gini_g-gini_c
            outs['PMP'] = blur_esti
            outs['BPR'] = blur_pred

        return outs


