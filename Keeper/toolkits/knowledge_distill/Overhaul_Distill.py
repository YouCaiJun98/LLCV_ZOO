'''Naive Pytorch implementation of Feature Distillation.
Reference: https://github.com/clovaai/overhaul-distillation
See the paper "A Comprehensive Overhaul of Feature Distillation"
for more details.
We made a little modification to the original overhaul FKD, supporting
naive FKD.
'''
import math
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverhaulDistillator(nn.Module):
    def __init__(self, t_model, s_model, pre_act=True, use_margin=False, bn_margin=False):
        super(OverhaulDistillator, self).__init__()
        self.t_model = t_model
        self.s_model = s_model
        self.pre_act = pre_act
        self.use_margin = use_margin
        self.bn_margin  = bn_margin

        t_chs = self.t_model.get_ch_num()
        s_chs = self.s_model.get_ch_num()
        self.transforms = nn.ModuleList([self.build_transformation(t, s) for (t, s) in zip(t_chs, s_chs)])

        if self.use_margin and self.bn_margin:
            t_bns = self.t_model.get_bn_before_act()
            margins = [get_margin_from_BN(bn) for bn in t_bns]
            for i, margin in enumerate(margins):
                self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())


    def forward(self, x):
        t_feats, t_out = self.t_model.extract_feature(x, PreAct=self.pre_act)
        s_feats, s_out = self.s_model.extract_feature(x, PreAct=self.pre_act)

        feat_num = len(t_feats)
        loss_distill = 0
        for i in range(feat_num):
            s_feats[i] = self.transforms[i](s_feats[i])
            if self.use_margin:
                margin = getattr(self, 'margin%d' % (i+1)) / 2 ** (feat_num - i - 1)
                loss_distill += self.distillation_loss(s_feats[i], t_feats[i].detach(), margin)
            else:
                loss_distill += self.distillation_loss(s_feats[i], t_feats[i].detach())

        return s_out, loss_distill


    def build_transformation(self, t_ch, s_ch):
        student_transform = [nn.Conv2d(s_ch, t_ch, kernel_size=1, stride=1, padding=0, bias=False),]
        if self.use_margin and self.bn_margin:
            student_transform.append(nn.BatchNorm2d(t_ch))

        # we initialize the 1*1 Conv and BN as original overhaul-distill does.
        for m in student_transform:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        return nn.Sequential(*student_transform)


    def get_margin_from_BN(bn):
        margin = []
        std = bn.weight.data
        mean = bn.bias.data
        for (s, m) in zip(std, mean):
            s = abs(s.item())
            m = m.item()
            if norm.cdf(-m / s) > 0.001:
                margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
            else:
                margin.append(-3 * s)

        return torch.FloatTensor(margin).to(std.device)


    def distillation_loss(self, s_feat, t_feat, margin=None):
        if self.use_margin:
            t_feat = torch.max(t_feat, margin)

        loss = F.mse_loss(s_feat, t_feat, reduction='none')

        if self.use_margin:
            loss = loss * ((s_feat > t_feat) | (t_feat > 0)).float()

        return loss.sum()


