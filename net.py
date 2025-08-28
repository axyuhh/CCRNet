import torch.nn as nn
import torch
from function import calc_mean_std
import random

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()
)

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def calc_mean(feat):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean

def nor_mean(feat):
    size = feat.size()
    mean = calc_mean(feat)
    nor_feat = feat - mean.expand(size)
    return nor_feat, mean

class adain(nn.Module):
    def __init__(self, in_planes):
        super(adain, self).__init__()
        self.f_san = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g_san = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h_san = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)

        self.f_mcc = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g_mcc = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.fc = nn.Linear(in_planes, in_planes)

        self.out_conv = nn.Conv2d(in_planes, in_planes, (1, 1))

        self.cnet = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 128, 1, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 32, 1, 1, 0))
        self.snet = nn.Sequential(nn.Conv2d(512, 256, 3, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(256, 128, 3, 1, 0),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 32, 1, 1, 0))
        self.uncompress = nn.Conv2d(32, 512, 1, 1, 0)

    def forward(self, content, style):
        B, C, H, W = content.size()
        F_Fc_norm = self.f_mcc(mean_variance_norm(content))
        G_Fs_norm = self.g_mcc(mean_variance_norm(style)).view(-1, 1, H * W)
        G_Fs_sum = G_Fs_norm.view(B, C, H * W).sum(-1)
        FC_S = torch.bmm(G_Fs_norm, G_Fs_norm.permute(0, 2, 1)).view(B, C) / G_Fs_sum
        FC_S = self.fc(FC_S).view(B, C, 1, 1)
        part1 = F_Fc_norm * FC_S

        Fc_norm = mean_variance_norm(content)
        Fs_norm, smean = nor_mean(style)
        Fc_norm = self.cnet(Fc_norm)
        Fs_norm = self.snet(Fs_norm)
        b, c, h, w = Fc_norm.size()
        Fs_cov = Fs_norm.flatten(2, 3)
        out = torch.bmm(Fs_cov, Fs_cov.permute(0, 2, 1)).div(Fs_cov.size(2))
        out = torch.bmm(out, Fc_norm.flatten(2, 3)).view(b, c, h, w)
        part2 = self.uncompress(out)

        F_san = self.f_san(mean_variance_norm(content))
        G_san = self.g_san(mean_variance_norm(style))
        H_san = self.h_san(style)
        F_san = F_san.view(B, -1, W * H).permute(0, 2, 1)
        G_san = G_san.view(B, -1, W * H)
        S_san = torch.bmm(F_san, G_san)
        S_san = self.sm(S_san)
        H_san = H_san.view(B, -1, W * H)
        Output = torch.bmm(H_san, S_san.permute(0, 2, 1))
        part3 = Output.view(B, C, H, W)

        part123 = part1 * part2 * part3
        part123 = torch.pow(torch.relu(part123), 1 / 3)
        return self.out_conv(part123) + content

class Net(nn.Module):
    def __init__(self, encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        self.decoder = decoder
        self.transform = adain(in_planes=512)
        self.mse_loss = nn.MSELoss()

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        liest = [[2], [3], [2, 3]]
        ran_listt = random.choice(liest)
        warp_content = torch.flip(content, ran_listt)

        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        warp_content_feats = self.encode_with_intermediate(warp_content)

        warp_t = self.transform(warp_content_feats[3], style_feats[3])
        warp_g_t = self.decoder(warp_t)

        t = self.transform(content_feats[3], style_feats[3])
        g_t = self.decoder(t)
        cc_t = self.transform(content_feats[3], content_feats[3])
        cc_g_t = self.decoder(cc_t)
        ss_t = self.transform(style_feats[3], style_feats[3])
        ss_g_t = self.decoder(ss_t)
        Fcc = self.encode_with_intermediate(cc_g_t)
        Fss = self.encode_with_intermediate(ss_g_t)
        l_identity1 = self.calc_content_loss(cc_g_t, content) + self.calc_content_loss(ss_g_t, style)
        l_identity2 = self.calc_content_loss(Fcc[0], content_feats[0]) + self.calc_content_loss(Fss[0], style_feats[0])
        for i in range(1, 4):
            l_identity2 += self.calc_content_loss(Fcc[i], content_feats[i]) + self.calc_content_loss(Fss[i], style_feats[i])

        warp_g_t_feats = self.encode_with_intermediate(warp_g_t)

        loss_warp = self.calc_content_loss(mean_variance_norm(warp_g_t_feats[-1]), mean_variance_norm(warp_content_feats[-1])) * 2.0
        loss_warp += self.calc_content_loss(warp_g_t, torch.flip(g_t, ran_listt).detach()) * 50.0
        for i in range(0, 4):
            loss_warp += self.calc_style_loss(warp_g_t_feats[i], style_feats[i]) * 4.0
        g_t_feats = self.encode_with_intermediate(g_t)
        loss_c = self.calc_content_loss(mean_variance_norm(g_t_feats[-1]), mean_variance_norm(content_feats[-1]))
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s, l_identity1, l_identity2, loss_warp
