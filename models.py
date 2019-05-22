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
        self.expression_pred = nn.Linear(2048, 3)
        self.face_prob_pred = nn.Linear(2048, 1)
        self.face_shape_pred = nn.Linear(2048, 5)
        self.face_type_pred = nn.Linear(2048, 2)
        self.gender_pred = nn.Linear(2048, 2)
        self.glasses_pred = nn.Linear(2048, 3)
        self.race_pred = nn.Linear(2048, 4)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # nn.init.xavier_uniform_(self.fc.weight)

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
        expression_out = self.softmax(self.expression_pred(x))
        face_prob_out = self.sigmoid(self.face_prob_pred(x))
        face_shape_out = self.softmax(self.face_shape_pred(x))
        face_type_out = self.softmax(self.face_type_pred(x))
        gender_out = self.softmax(self.gender_pred(x))
        glasses_out = self.softmax(self.glasses_pred(x))
        race_out = self.softmax(self.race_pred(x))
        return age_out, pitch_out, roll_out, yaw_out, beauty_out, expression_out, face_prob_out, face_shape_out, face_type_out, gender_out, glasses_out, race_out


if __name__ == "__main__":
    model = FaceAttributesModel().to(device)
    summary(model, (3, 224, 224))
