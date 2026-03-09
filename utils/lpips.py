import torch
import torch.nn as nn
from torchvision.models import alexnet
from collections import OrderedDict

class LPIPS(nn.Module):
    def __init__(self, net='alex', version='0.1', spatial=False, lpips=True):
        super(LPIPS, self).__init__()
        self.spatial = spatial
        self.lpips = lpips
        self.version = version

        # Load AlexNet
        self.net = AlexNet()
        self.net.load_state_dict(torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', progress=False))
        
        self.chns = [64, 192, 384, 256, 256]
        self.lins = nn.ModuleList()
        for i in range(len(self.chns)):
            self.lins.append(NetLin(self.chns[i], use_dropout=True))
        
        self.lins.load_state_dict(torch.hub.load_state_dict_from_url('https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/models/weights/v0.1/alex.pth', progress=False))
        
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, in0, in1):
        in0_input, in1_input = (in0 * 2 - 1), (in1 * 2 - 1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        
        feats0, feats1, diffs = {}, {}, {}
        for i in range(len(self.chns)):
            feats0[i], feats1[i] = normalize_tensor(outs0[i]), normalize_tensor(outs1[i])
            diffs[i] = (feats0[i] - feats1[i]) ** 2
        
        res = [spatial_average(self.lins[i].model(diffs[i]), keepdim=True) for i in range(len(self.chns))]
        val = sum(res)
        
        if self.spatial:
            return val
        else:
            return val.mean()

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        alexnet_pretrained = alexnet(pretrained=False)
        self.features = alexnet_pretrained.features
        self.avgpool = alexnet_pretrained.avgpool
        self.classifier = nn.Sequential(*list(alexnet_pretrained.classifier.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return [x] # a list of feature maps

class NetLin(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLin, self).__init__()
        layers = [nn.Dropout(),] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3], keepdim=keepdim) 