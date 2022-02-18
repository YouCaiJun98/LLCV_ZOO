import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gumbelmodule import GumbelSoftmax


__all__ = ['D_UNet', 'd_unet', 'ActivationRateAccum']
class D_UNet(nn.Module):
    # a thin implementation
    def __init__(self, inchannel=3, scale=1):
        super(D_UNet, self).__init__()
        self.scale = scale # channel inflation

        self.conv1_1 = nn.Conv2d(inchannel, 16*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(16*self.scale, 16*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.gate_1 = GateUnit(16*self.scale)

        self.conv2_1 = nn.Conv2d(16*self.scale, 32*self.scale, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(32*self.scale, 32*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.gate_2 = GateUnit(32*self.scale)

        self.conv3_1 = nn.Conv2d(32*self.scale, 64*self.scale, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(64*self.scale, 64*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.gate_3 = GateUnit(64*self.scale)

        self.conv4_1 = nn.Conv2d(64*self.scale, 128*self.scale, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(128*self.scale, 128*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.gate_4 = GateUnit(128*self.scale)

        self.conv5_1 = nn.Conv2d(128*self.scale, 256*self.scale, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(256*self.scale, 256*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

        self.up6 = nn.ConvTranspose2d(256*self.scale, 128*self.scale, 2, stride=2, bias=False)
        self.conv6_1 = nn.Conv2d(256*self.scale, 128*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6_2 = nn.Conv2d(128*self.scale, 128*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

        self.up7 = nn.ConvTranspose2d(128*self.scale, 64*self.scale, 2, stride=2, bias=False)
        self.conv7_1 = nn.Conv2d(128*self.scale, 64*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7_2 = nn.Conv2d(64*self.scale, 64*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

        self.up8 = nn.ConvTranspose2d(64*self.scale, 32*self.scale, 2, stride=2, bias=False)
        self.conv8_1 = nn.Conv2d(64*self.scale, 32*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8_2 = nn.Conv2d(32*self.scale, 32*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

        self.up9 = nn.ConvTranspose2d(32*self.scale, 16*self.scale, 2, stride=2, bias=False)
        self.conv9_1 = nn.Conv2d(32*self.scale, 16*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv9_2 = nn.Conv2d(16*self.scale, 16*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv10 = nn.Conv2d(16*self.scale, inchannel, kernel_size=1, stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x):
        gate_activations = []

        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x
        choice_1 = self.gate_1(x)
        gate_activations.append(choice_1[:, 1])

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x
        choice_2 = self.gate_2(x)
        gate_activations.append(choice_2[:, 1])

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x
        choice_3 = self.gate_3(x)
        gate_activations.append(choice_3[:, 1])

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x
        choice_4 = self.gate_4(x)
        gate_activations.append(choice_4[:, 1])

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)

        x = (self.up6(x) * choice_4[:, 1].unsqueeze(1))[:, :, :conv4.size(2), :conv4.size(3)] \
                + conv4 * choice_4[:, 0].unsqueeze(1)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        #x = torch.cat((x, conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = (self.up7(x) * choice_3[:, 1].unsqueeze(1))[:, :, :conv3.size(2), :conv3.size(3)] \
                + conv3 * choice_3[:, 0].unsqueeze(1)
        #x = torch.cat((x, conv3), 1)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)
        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = (self.up8(x) * choice_2[:, 1].unsqueeze(1))[:, :, :conv2.size(2), :conv2.size(3)] \
                + conv2 * choice_2[:, 0].unsqueeze(1)
        #x = torch.cat((x, conv2), 1)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)
        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = (self.up9(x) * choice_1[:, 1].unsqueeze(1))[:, :, :conv1.size(2), :conv1.size(3)] \
                + conv1 * choice_1[:, 0].unsqueeze(1)
        #x = torch.cat((x, conv1), 1)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)
        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)

        return x, gate_activations

class GateUnit(nn.Module):
    def __init__(self, in_planes, temp=1, scale=1):
        super(GateUnit, self).__init__()
        self.scale = scale
        self.temp = temp
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, 16*self.scale, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc1bn = nn.BatchNorm2d(16*self.scale)
        self.fc2 = nn.Conv2d(16*self.scale, 2, kernel_size=1)
        # initialization from Conv-AIG
        self.fc2.bias.data[0] = 0.1
        self.fc2.bias.data[1] = 2
        # self.gs = GumbelSoftmax()

    def forward(self, x):
        # Compute relevance score
        #import ipdb; ipdb.set_trace()
        score = self.avgpool(x)
        score = F.relu(self.fc1bn(self.fc1(score)))
        score = self.fc2(score)
        # Sample from Gumbel Module
        choice = F.gumbel_softmax(score, tau=self.temp, hard=True, dim=1)
        #choice = self.gs(score, temp=self.temp, force_hard=True)

        return choice

def d_unet(**kwargs):
    model = D_UNet(**kwargs)
    return model

class ActivationRateAccum():
    def __init__(self, gate_num):
        self.gates = {i: 0 for i in range(gate_num)}
        self.batch_num = 0
        self.batch_size = 0

    def accumulate(self, activations):
        for i, act in enumerate(activations):
            self.gates[i] += torch.sum(act)
        
        self.batch_size = activations[0].size(0) # activations should be a list
        self.batch_num += 1

    def getoutput(self):
        data_num = self.batch_size * self.batch_num
        return {i: self.gates[i].data.cpu().numpy() / (data_num) for i in self.gates}
