from torch import nn
from torchsummary import summary
from torchvision import models

from config import device


class FaceAttributesModel(nn.Module):
    def __init__(self):
        super(FaceAttributesModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(2048)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 4, 4]
        x = self.bn(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 2048]
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model = FaceAttributesModel().to(device)
    summary(model, (3, 224, 224))
