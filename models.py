from torch import nn
from torch.nn import functional   as F
from torchsummary import summary
from torchvision import models

from config import device


class FaceAttributesModel(nn.Module):
    def __init__(self):
        super(FaceAttributesModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.avg_pool = nn.AvgPool2d(4)
        self.conv = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        # self.age_fc = nn.Linear(512, 512)
        # self.age_pred = nn.Linear(512, 1)
        # self.pitch_fc = nn.Linear(512, 512)
        # self.pitch_pred = nn.Linear(512, 1)
        # self.roll_fc = nn.Linear(512, 512)
        # self.roll_pred = nn.Linear(512, 1)
        # self.yaw_fc = nn.Linear(512, 512)
        # self.yaw_pred = nn.Linear(512, 1)
        self.beauty_fc = nn.Linear(512, 512)
        self.beauty_pred = nn.Linear(512, 1)
        # self.expression_fc = nn.Linear(512, 512)
        # self.expression_pred = nn.Linear(512, 3)
        # self.face_prob_fc = nn.Linear(512, 512)
        # self.face_prob_pred = nn.Linear(512, 1)
        # self.face_shape_fc = nn.Linear(512, 512)
        # self.face_shape_pred = nn.Linear(512, 5)
        # self.face_type_fc = nn.Linear(512, 512)
        # self.face_type_pred = nn.Linear(512, 2)
        # self.gender_fc = nn.Linear(512, 512)
        # self.gender_pred = nn.Linear(512, 2)
        # self.glasses_fc = nn.Linear(512, 512)
        # self.glasses_pred = nn.Linear(512, 3)
        # self.race_fc = nn.Linear(512, 512)
        # self.race_pred = nn.Linear(512, 4)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1)

        # nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 4, 4]
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [N, 512]
        # age_out = self.sigmoid(self.age_pred(F.relu(self.age_fc(x))))
        # pitch_out = self.sigmoid(self.pitch_pred(F.relu(self.pitch_fc(x))))
        # roll_out = self.sigmoid(self.roll_pred(F.relu(self.roll_fc(x))))
        # yaw_out = self.sigmoid(self.yaw_pred(F.relu(self.yaw_fc(x))))
        beauty_out = self.sigmoid(self.beauty_pred(F.relu(self.beauty_fc(x))))
        # expression_out = self.softmax(self.expression_pred(F.relu(self.expression_fc(x))))
        # face_prob_out = self.sigmoid(self.face_prob_pred(F.relu(self.face_prob_fc(x))))
        # face_shape_out = self.softmax(self.face_shape_pred(F.relu(self.face_shape_fc(x))))
        # face_type_out = self.softmax(self.face_type_pred(F.relu(self.face_type_fc(x))))
        # gender_out = self.softmax(self.gender_pred(F.relu(self.gender_fc(x))))
        # glasses_out = self.softmax(self.glasses_pred(F.relu(self.glasses_fc(x))))
        # race_out = self.softmax(self.race_pred(F.relu(self.race_fc(x))))
        # return age_out, pitch_out, roll_out, yaw_out, beauty_out, expression_out, face_prob_out, face_shape_out, face_type_out, gender_out, glasses_out, race_out
        return beauty_out


if __name__ == "__main__":
    model = FaceAttributesModel().to(device)
    summary(model, (3, 112, 112))
