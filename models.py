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
        self.age_pred = nn.Linear(2048, 1)
        self.pitch_pred = nn.Linear(2048, 1)
        self.roll_pred = nn.Linear(2048, 1)
        self.yaw_pred = nn.Linear(2048, 1)
        self.beauty_pred = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 4, 4]
        x = self.bn(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 2048]
        age_out = self.sigmoid(self.age_pred(x))
        pitch_out = self.sigmoid(self.pitch_pred(x))
        roll_out = self.sigmoid(self.roll_pred(x))
        yaw_out = self.sigmoid(self.yaw_pred(x))
        beauty_out = self.sigmoid(self.beauty_pred(x))
        return age_out, pitch_out, roll_out, yaw_out, beauty_out


if __name__ == "__main__":
    model = FaceAttributesModel().to(device)
    summary(model, (3, 224, 224))
