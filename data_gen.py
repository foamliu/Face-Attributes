import pickle

from torch.utils.data import Dataset
from torchvision import transforms

from config import pickle_file_landmarks, num_train
from utils import align_face

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class FaceAttributesDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file_landmarks, 'rb') as file:
            data = pickle.load(file)

        samples = data['samples']

        if split == 'train':
            self.samples = samples[:num_train]
        else:
            self.samples = samples[num_train:]

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        full_path = sample['full_path']
        landmarks = sample['landmarks']

        try:
            img = align_face(full_path, landmarks)
        except Exception:
            print('full_path: ' + full_path)
            raise

        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        label = sample['attr']['beauty'] / 100.
        return img, label

    def __len__(self):
        return len(self.samples)
