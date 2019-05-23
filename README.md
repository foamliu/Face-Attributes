# Face-Attributes

Deep Face Attributes.


## DataSet

CASIA WebFace DataSet, 479,653 faces.

### Gender

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/gender_dist.png)

### Age

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/age_dist.png)

### Euler angles:

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/euler_angles.png)

Pitch:

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/angle_pitch_dist.png)

Yaw:

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/angle_yaw_dist.png)

Roll:

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/angle_roll_dist.png)

### Beauty

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/beauty_dist.png)

### Expression

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/expression_dist.png)

### Face shape

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/face_shape_dist.png)

### Face type

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/face_type_dist.png)

### Glasses

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/glasses_dist.png)

### Race

![image](https://github.com/foamliu/Face-Attributes/raw/master/images/race_dist.png)

## Dependencies
- PyTorch 1.0.0

## Usage


### Train
```bash
$ python train.py
```

To visualize the training process：
```bash
$ tensorboard --logdir=runs
```

Image | Aligned | Out | True |
|---|---|---|---|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/0_img.jpg)|age: 32
pitch: 8.26
roll: -4.09
yaw: -0.95
beauty: 48.75
expression: none
face_prob: 0.7677
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|age: 36
pitch: 8.13
roll: -8.54
yaw: 42.44
beauty: 47.49
expression: none
face_prob: 0.86
face_shape: oval
face_type: human
gender: female
glasses: common
race: white
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/1_img.jpg)|age: 32
pitch: 8.07
roll: -3.99
yaw: -1.41
beauty: 48.75
expression: none
face_prob: 0.7648
face_shape: oval
face_type: human
gender: female
glasses: none
race: white
|age: 35
pitch: 7.83
roll: 6.93
yaw: 17.39
beauty: 40.61
expression: none
face_prob: 1
face_shape: oval
face_type: human
gender: female
glasses: common
race: white
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/2_img.jpg)|age: 32
pitch: 8.32
roll: -4.07
yaw: -0.82
beauty: 48.75
expression: none
face_prob: 0.7687
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|age: 37
pitch: 1.89
roll: 4.37
yaw: 21.98
beauty: 29.69
expression: none
face_prob: 0.95
face_shape: round
face_type: human
gender: male
glasses: none
race: white
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/3_img.jpg)|age: 32
pitch: 8.32
roll: -4.09
yaw: -0.81
beauty: 48.75
expression: none
face_prob: 0.7686
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|age: 23
pitch: 10.95
roll: -14.12
yaw: -3.75
beauty: 48
expression: none
face_prob: 0.87
face_shape: oval
face_type: human
gender: male
glasses: none
race: black
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/4_img.jpg)|age: 32
pitch: 8.09
roll: -4.11
yaw: -1.41
beauty: 48.76
expression: none
face_prob: 0.7648
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|age: 28
pitch: 21.48
roll: -2.47
yaw: -3.53
beauty: 42.99
expression: smile
face_prob: 1
face_shape: oval
face_type: human
gender: female
glasses: none
race: white
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/5_img.jpg)|age: 32
pitch: 8.32
roll: -4.09
yaw: -0.8
beauty: 48.75
expression: none
face_prob: 0.7687
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|age: 23
pitch: 8.42
roll: -17.64
yaw: 77.12
beauty: 31.74
expression: none
face_prob: 0.96
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/6_img.jpg)|age: 32
pitch: 8.16
roll: -4.07
yaw: -1.21
beauty: 48.75
expression: none
face_prob: 0.7661
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|age: 31
pitch: 23.92
roll: -15.42
yaw: 19.38
beauty: 33.34
expression: none
face_prob: 0.95
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/7_img.jpg)|age: 32
pitch: 7.47
roll: -4.43
yaw: -2.88
beauty: 48.76
expression: smile
face_prob: 0.7558
face_shape: oval
face_type: human
gender: female
glasses: none
race: white
|age: 35
pitch: 18.75
roll: -3.63
yaw: -0.26
beauty: 58.88
expression: smile
face_prob: 0.98
face_shape: oval
face_type: human
gender: female
glasses: none
race: white
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/8_img.jpg)|age: 32
pitch: 7.98
roll: -4.3
yaw: -1.7
beauty: 48.76
expression: none
face_prob: 0.7633
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|age: 25
pitch: 20.34
roll: -2.22
yaw: -6.85
beauty: 64.57
expression: smile
face_prob: 1
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/9_img.jpg)|age: 32
pitch: 8.27
roll: -4.08
yaw: -0.93
beauty: 48.75
expression: none
face_prob: 0.7679
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|age: 36
pitch: 21.82
roll: 7.21
yaw: 10.6
beauty: 68.71
expression: none
face_prob: 1
face_shape: oval
face_type: human
gender: male
glasses: none
race: white
|