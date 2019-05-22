import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
image_w = 224
image_h = 224
channel = 3

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
beauty_num_classes = 101
num_samples = 479653
train_ratio = 0.9
num_train = int(num_samples * train_ratio)
DATA_DIR = 'data'
IMG_DIR = 'data/CASIA-WebFace'
pickle_file = DATA_DIR + '/' + 'CASIA-WebFace.pkl'
pickle_file_landmarks = DATA_DIR + '/' + 'CASIA-WebFace-landmarks.pkl'
