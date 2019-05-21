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
        self.bn = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(2048 * 4 * 4, 2048)

        self.fc2 = nn.Linear(2048, beauty_num_classes)
        self.softmax = nn.Softmax(dim=-1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 4, 4]
        x = self.bn(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = F.relu(x)  # [N, 2048]
        x = self.fc2(x)  # [N, 101]
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = FaceAttributesModel().to(device)
    summary(model, (3, 112, 112))
