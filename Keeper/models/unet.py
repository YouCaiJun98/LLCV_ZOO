import torch
import torch.nn as nn
import math


__all__ = ['U_Net', 'unet']
class U_Net(nn.Module):
    # a thin implementation
    def __init__(self, inchannel=3, scale=1):
        super(U_Net, self).__init__()
        self.scale = scale # channel inflation

        self.conv1_1 = nn.Conv2d(inchannel, 16*self.scale, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu =  nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1_2 = nn.Conv2d(16*self.scale, 16*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv2_1 = nn.Conv2d(16*self.scale, 32*self.scale, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(32*self.scale, 32*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv3_1 = nn.Conv2d(32*self.scale, 64*self.scale, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(64*self.scale, 64*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(64*self.scale, 128*self.scale, kernel_size=3, stride=2, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(128*self.scale, 128*self.scale, kernel_size=3, stride=1, padding=1, bias=True)

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
        x = self.conv1_1(x)
        x = self.lrelu(x)
        x = self.conv1_2(x)
        x = self.lrelu(x)
        conv1 = x

        x = self.conv2_1(x)
        x = self.lrelu(x)
        x = self.conv2_2(x)
        x = self.lrelu(x)
        conv2 = x

        x = self.conv3_1(x)
        x = self.lrelu(x)
        x = self.conv3_2(x)
        x = self.lrelu(x)
        conv3 = x

        x = self.conv4_1(x)
        x = self.lrelu(x)
        x = self.conv4_2(x)
        x = self.lrelu(x)
        conv4 = x

        x = self.conv5_1(x)
        x = self.lrelu(x)
        x = self.conv5_2(x)
        x = self.lrelu(x)

        x = self.up6(x)
        x = torch.cat((x[:, :, :conv4.size(2), :conv4.size(3)], conv4), 1)
        #x = torch.cat((x, conv4), 1)
        x = self.conv6_1(x)
        x = self.lrelu(x)
        x = self.conv6_2(x)
        x = self.lrelu(x)

        x = self.up7(x)
        #x = torch.cat((x, conv3), 1)
        x = torch.cat((x[:, :, :conv3.size(2), :conv3.size(3)], conv3), 1)
        x = self.conv7_1(x)
        x = self.lrelu(x)
        x = self.conv7_2(x)
        x = self.lrelu(x)

        x = self.up8(x)
        #x = torch.cat((x, conv2), 1)
        x = torch.cat((x[:, :, :conv2.size(2), :conv2.size(3)], conv2), 1)
        x = self.conv8_1(x)
        x = self.lrelu(x)
        x = self.conv8_2(x)
        x = self.lrelu(x)

        x = self.up9(x)
        #x = torch.cat((x, conv1), 1)
        x = torch.cat((x[:, :, :conv1.size(2), :conv1.size(3)], conv1), 1)
        x = self.conv9_1(x)
        x = self.lrelu(x)
        x = self.conv9_2(x)
        x = self.lrelu(x)

        x = self.conv10(x)

        return x
    
def unet(**kwargs):
    model = U_Net(**kwargs)
    return model


