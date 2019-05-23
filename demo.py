import json
import pickle
import random

import cv2 as cv
import numpy as np

from config import *
from utils import align_face, idx2name

if __name__ == "__main__":
    with open(pickle_file_landmarks, 'rb') as file:
        data = pickle.load(file)

    samples = data['samples']

    num_samples = len(samples)
    num_train = int(train_ratio * num_samples)
    samples = samples[num_train:]

    samples = random.sample(samples, 10)

    inputs = torch.zeros([10, 3, image_h, image_w], dtype=torch.float, device=device)

    sample_preds = []

    for i, sample in enumerate(samples):
        full_path = sample['full_path']
        landmarks = sample['landmarks']
        print(full_path)
        raw = cv.imread(full_path)
        raw = cv.resize(raw, (image_w, image_h))
        filename = 'images/{}_raw.jpg'.format(i)
        cv.imwrite(filename, raw)
        img = align_face(full_path, landmarks)
        filename = 'images/{}_img.jpg'.format(i)
        cv.imwrite(filename, img)
        img = img.transpose(2, 0, 1)
        assert img.shape == (3, image_h, image_w)
        assert np.max(img) <= 255
        inputs[i] = torch.FloatTensor(img / 255.)

        age = sample['attr']['age']
        pitch = sample['attr']['angle']['pitch']
        roll = sample['attr']['angle']['roll']
        yaw = sample['attr']['angle']['yaw']
        beauty = sample['attr']['beauty']
        expression = sample['attr']['expression']['type']
        face_prob = sample['attr']['face_probability']
        face_shape = sample['attr']['face_shape']['type']
        face_type = sample['attr']['face_type']['type']
        gender = sample['attr']['gender']['type']
        glasses = sample['attr']['glasses']['type']
        race = sample['attr']['race']['type']
        sample_preds.append({'i': i, 'age_true': age,
                             'pitch_true': pitch,
                             'roll_true': roll,
                             'yaw_true': yaw,
                             'beauty_true': beauty,
                             'expression_true': expression,
                             'face_prob_true': face_prob,
                             'face_shape_true': face_shape,
                             'face_type_true': face_type,
                             'gender_true': gender,
                             'glasses_true': glasses,
                             'race_true': race})

    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(inputs)

    age_out, pitch_out, roll_out, yaw_out, beauty_out, expression_out, face_prob_out, face_shape_out, face_type_out, gender_out, glasses_out, race_out = output

    _, expression_out = expression_out.topk(1, 1, True, True)
    _, face_shape_out = face_shape_out.topk(1, 1, True, True)
    _, face_type_out = face_type_out.topk(1, 1, True, True)
    _, gender_out = gender_out.topk(1, 1, True, True)
    _, glasses_out = glasses_out.topk(1, 1, True, True)
    _, race_out = race_out.topk(1, 1, True, True)

    age_out = age_out.cpu().numpy()
    pitch_out = pitch_out.cpu().numpy()
    roll_out = roll_out.cpu().numpy()
    yaw_out = yaw_out.cpu().numpy()
    beauty_out = beauty_out.cpu().numpy()
    expression_out = expression_out.cpu().numpy()
    face_prob_out = face_prob_out.cpu().numpy()
    face_shape_out = face_shape_out.cpu().numpy()
    face_type_out = face_type_out.cpu().numpy()
    gender_out = gender_out.cpu().numpy()
    glasses_out = glasses_out.cpu().numpy()
    race_out = race_out.cpu().numpy()

    for i in range(10):
        sample = sample_preds[i]

        sample['age_out'] = int(age_out[i][0] * 100)
        sample['pitch_out'] = float('{0:.2f}'.format(pitch_out[i][0] * 360 - 180))
        sample['roll_out'] = float('{0:.2f}'.format(roll_out[i][0] * 360 - 180))
        sample['yaw_out'] = float('{0:.2f}'.format(yaw_out[i][0] * 360 - 180))
        sample['beauty_out'] = float('{0:.2f}'.format(beauty_out[i][0] * 100))
        sample['expression_out'] = idx2name(int(expression_out[i][0]), 'expression')
        sample['face_prob_out'] = float('{0:.4f}'.format(face_prob_out[i][0]))
        sample['face_shape_out'] = idx2name(int(face_shape_out[i][0]), 'face_shape')
        sample['face_type_out'] = idx2name(int(face_type_out[i][0]), 'face_type')
        sample['gender_out'] = idx2name(int(gender_out[i][0]), 'gender')
        sample['glasses_out'] = idx2name(int(glasses_out[i][0]), 'glasses')
        sample['race_out'] = idx2name(int(race_out[i][0]), 'race')

    with open('sample_preds.json', 'w') as file:
        json.dump(sample_preds, file, indent=4, ensure_ascii=False)
