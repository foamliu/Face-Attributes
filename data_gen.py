import os
import pickle

import cv2 as cv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR, pickle_file, num_train

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class FaceAttributesDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        samples = data['samples']

        if split == 'train':
            self.samples = samples[:num_train]
        else:
            self.samples = samples[num_train:]

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['full_path']
        label = sample['label']

        img = cv.imread(filename)  # BGR
        img = img[..., ::-1]  # RGB
        img = Image.fromarray(img, 'RGB')  # RGB
        img = self.transformer(img)  # RGB

        return img, label

    def __len__(self):
        return len(self.samples)
