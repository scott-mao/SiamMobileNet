from __future__ import absolute_import
import math
import torch.nn as nn
import torch
from pytorch.pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable
from thop import profile
import numpy as np
# class fuse_layer(nn.Module):
#     def __init__(self):
#         super(fuse_layer,self).__init__()
#         self.fuse = nn.Sequential(nn.Conv2d(2,1,kernel_size=1,stride=1,padding=0))
#
#     def forward(self,x):
#         out = self.fuse(x)
#         return out

######   bacobone_nochage + crop

class Mobilenet_original(nn.Module):
    def __init__(self,k_size = 3):
        super(Mobilenet_original, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.layer1 = conv_bn(3, 32, 2)
        self.layer2 = conv_dw(32, 64, 1)
        self.layer3 = conv_dw(64, 128, 2)
        self.layer4 = conv_dw(128, 128, 1)
        self.layer5 = conv_dw(128, 256, 2)#2
        self.layer6 = conv_dw(256, 256, 1)
        self.layer7 = conv_dw(256, 512, 1)#2
        self.layer8 = conv_dw(512, 512, 1)
        self.layer9 = conv_dw(512, 512, 1)
        # self.layer10 = conv_dw(512, 512, 1)
        # self.layer11 = conv_dw(512, 512, 1)
        # self.layer12 = conv_dw(512, 512, 1)
        self.layer10 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()
        self._load_ini_weights()

    def crop(self,x):
        return x[:,:,1:-1,1:-1].contiguous()

    def forward(self, x,is_z = False):
        x = self.layer1(x)
        # x = self.crop(x)
        x = self.layer2(x)
        x = self.crop(x)
        x = self.layer3(x)
        # x = self.crop(x)
        x = self.layer4(x)
        x = self.crop(x)
        x = self.layer5(x)
        # x = self.crop(x)
        x = self.layer6(x)
        x2 = self.crop(x)
        x = self.layer7(x2)
        x = self.crop(x)
        x = self.layer8(x)
        x = self.crop(x)
        x = self.layer9(x)
        x = self.crop(x)

        x3 = self.layer10(x)
        # # x = self.crop(x)
        # x = self.layer11(x)
        # # x = self.crop(x)
        # x = self.layer12(x)
        # # x = self.crop(x)
        if is_z:
            b5,c5,h5,w5 = x3.size()
            y3 = self.avg_pool(x3)# (8,256,1,1)
            y3 = self.conv(y3.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
            y3 = self.sigmoid(y3)#  (8,256,1,1)
            x3 = x3 * y3.expand_as(x3) + x3
            # xa = self.avg_pool(x3).cpu().detach().numpy().squeeze(0).squeeze(-1).squeeze(-1)
            # np.savetxt("a.txt" , xa)

            # xna = self.avg_pool(x).cpu().detach().numpy().squeeze(0).squeeze(-1).squeeze(-1)
            # np.savetxt("na.txt", xna)
        return x3



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _load_ini_weights(self):
        pre_model = ptcv_get_model("mobilenet_w1", pretrained=True)
        m0 = list(pre_model.modules())
        # print(m0)
        m1 = list(self.modules())
        # print(m1)
        count = 0
        for m in self.modules():
            if count == 2:
                m.weight.data = m0[3].weight.data
            if count == 6:
                m.weight.data = m0[9].weight.data
            if count == 9:
                m.weight.data = m0[13].weight.data
            if count == 13:
                m.weight.data = m0[19].weight.data
            if count == 16:
                m.weight.data = m0[23].weight.data
            if count == 20:
                m.weight.data = m0[28].weight.data
            if count == 23:
                m.weight.data = m0[32].weight.data
            if count == 27:
                m.weight.data = m0[38].weight.data
            if count == 30:
                m.weight.data = m0[42].weight.data
            if count == 34:
                m.weight.data = m0[47].weight.data
            if count == 37:
                m.weight.data = m0[51].weight.data
            if count == 41:
                m.weight.data = m0[57].weight.data
            if count == 44:
                m.weight.data = m0[61].weight.data
            if count == 48:
                m.weight.data = m0[66].weight.data
            if count == 51:
                m.weight.data = m0[70].weight.data
            if count == 55:
                m.weight.data = m0[75].weight.data
            if count == 58:
                m.weight.data = m0[79].weight.data

            count += 1






if __name__ == "__main__":
    model = Mobilenet_original()
    input = torch.randn(1, 3, 255, 255)
    flops, params = profile(model, inputs=(input,))
    # x = torch.randn(1, 3, 127, 127)
    # model(x)
    # print(model(x).shape)
    # print((model))
    print(params)



