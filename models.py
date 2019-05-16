import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torch import nn
from torch.nn import Parameter
from torchsummary import summary
from torchvision import transforms

from config import device, num_classes
from utils import parse_args


class ArcFaceModel50(nn.Module):
    def __init__(self, args):
        super(ArcFaceModel50, self).__init__()

        resnet = resnet50(pretrained=args.pretrained)

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.bn1 = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048 * 4 * 4, args.emb_size)
        self.bn2 = nn.BatchNorm1d(args.emb_size)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 4, 4]
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 512]
        x = self.fc(x)
        x = self.bn2(x)
        return x