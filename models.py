import torch
from torch import nn

from config import device

checkpoint = 'BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model'].to(device)
model.eval()


class FaceAttributesModel(nn.Module):
    def __init__(self):
        super(FaceAttributesModel, self).__init__()

        self.fc = nn.Linear(512, 1)

    def forward(self, images):
        x = model(images)
        x = self.fc(x)
        return x
