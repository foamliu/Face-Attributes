# -*- coding: utf-8 -*-
import json


def get_attrs(item, split):
    age = item['age_' + split]
    pitch = item['pitch_' + split]
    roll = item['roll_' + split]
    yaw = item['yaw_' + split]
    beauty = item['beauty_' + split]
    expression = item['expression_' + split]
    face_prob = item['face_prob_' + split]
    face_shape = item['face_shape_' + split]
    face_type = item['face_type_' + split]
    gender = item['gender_' + split]
    glasses = item['glasses_' + split]
    race = item['race_' + split]
    result = 'age: {}\n'.format(age)
    result += 'pitch: {}\n'.format(pitch)
    result += 'roll: {}\n'.format(roll)
    result += 'yaw: {}\n'.format(yaw)
    result += 'beauty: {}\n'.format(beauty)
    result += 'expression: {}\n'.format(expression)
    result += 'face_prob: {}\n'.format(face_prob)
    result += 'face_shape: {}\n'.format(face_shape)
    result += 'face_type: {}\n'.format(face_type)
    result += 'gender: {}\n'.format(gender)
    result += 'glasses: {}\n'.format(glasses)
    result += 'race: {}\n'.format(race)
    return result


if __name__ == '__main__':
    with open('README.t', 'r', encoding="utf-8") as file:
        text = file.readlines()
    text = ''.join(text)

    with open('sample_preds.json', 'r', encoding="utf-8") as file:
        results = json.load(file)

    for i in range(10):
        item = results[i]
        result_true = get_attrs(item, 'true')
        result_out = get_attrs(item, 'out')
        text = text.replace('$(result_true_{})'.format(i), result_true)
        text = text.replace('$(result_out_{})'.format(i), result_out)

    with open('README.md', 'w', encoding="utf-8") as file:
        file.write(text)
