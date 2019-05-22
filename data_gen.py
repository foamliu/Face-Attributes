import pickle

from torch.utils.data import Dataset
from torchvision import transforms

from config import pickle_file_landmarks, num_train
from utils import align_face


def name2idx(name):
    lookup_table = {'none': 0, 'smile': 1, 'laugh': 2,
                    'square': 0, 'oval': 1, 'heart': 2, 'round': 3, 'triangle': 4,
                    'human': 0, 'cartoon': 1,
                    'female': 0, 'male': 1,
                    'sun': 1, 'common': 2,
                    'yellow': 0, 'white': 1, 'black': 2, 'arabs': 3}

    return lookup_table[name]


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

        age = sample['attr']['age'] / 100.
        pitch = (sample['attr']['angle']['pitch'] + 180) / 360
        roll = (sample['attr']['angle']['roll'] + 180) / 360
        yaw = (sample['attr']['angle']['yaw'] + 180) / 360
        beauty = sample['attr']['beauty'] / 100.
        expression = name2idx(sample['attr']['expression']['type'])
        face_prob = sample['attr']['face_probability']
        face_shape = name2idx(sample['attr']['face_shape']['type'])
        face_type = name2idx(sample['attr']['face_type']['type'])
        gender = name2idx(sample['attr']['gender']['type'])
        glasses = name2idx(sample['attr']['glasses']['type'])
        race = name2idx(sample['attr']['race']['type'])
        return img, (age, pitch, roll, yaw, beauty, expression, face_prob, face_shape, face_type, gender, glasses, race)

    def __len__(self):
        return len(self.samples)
