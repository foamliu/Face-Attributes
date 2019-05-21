import torch.nn.functional as F
from torch import nn
from torchsummary import summary
from torchvision import models

from config import device, beauty_num_classes


class FaceAttributesModel(nn.Module):
    def __init__(self):
        super(FaceAttributesModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, 512)
        self.beauty_pred = nn.Linear(512, beauty_num_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.beauty_pred.weight)

    def forward(self, images):
        x = self.resnet(images)  # [N, 512, 1, 1]
        x = self.pool(x)
        x = x.view(-1, 512)  # [N, 512]

        beauty_out = F.relu(self.fc1(x))  # [N, 512]
        beauty_out = self.age_pred(beauty_out)  # [N, 101]
        return beauty_out


if __name__ == "__main__":
    model = FaceAttributesModel().to(device)
    summary(model, (3, 112, 112))
