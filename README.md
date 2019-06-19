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
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 31<br>pitch: 13.8<br>roll: -22.0<br>yaw: 37.96<br>beauty: 47.97<br>expression: none<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: female<br>glasses: none<br>race: white|age: 32<br>pitch: 15.17<br>roll: -23.43<br>yaw: 44.03<br>beauty: 45.12<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 35<br>pitch: 6.48<br>roll: 2.68<br>yaw: -36.36<br>beauty: 53.43<br>expression: smile<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: male<br>glasses: none<br>race: white|age: 33<br>pitch: 6.07<br>roll: 2.06<br>yaw: -43.75<br>beauty: 47.86<br>expression: smile<br>face_prob: 0.98<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 32<br>pitch: 11.0<br>roll: -1.31<br>yaw: -9.15<br>beauty: 60.07<br>expression: none<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: male<br>glasses: none<br>race: white|age: 32<br>pitch: 12.18<br>roll: -1.89<br>yaw: -7.93<br>beauty: 66.93<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 46<br>pitch: 25.04<br>roll: 16.2<br>yaw: -6.6<br>beauty: 52.13<br>expression: none<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: male<br>glasses: none<br>race: white|age: 48<br>pitch: 26.25<br>roll: 14.17<br>yaw: -2.81<br>beauty: 43.47<br>expression: none<br>face_prob: 0.84<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 36<br>pitch: 9.13<br>roll: 18.15<br>yaw: -27.27<br>beauty: 60.59<br>expression: none<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: male<br>glasses: none<br>race: white|age: 34<br>pitch: 8.15<br>roll: 19.08<br>yaw: -29.01<br>beauty: 68.18<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 36<br>pitch: -8.43<br>roll: 5.06<br>yaw: -20.05<br>beauty: 50.04<br>expression: none<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: male<br>glasses: none<br>race: white|age: 37<br>pitch: -9.41<br>roll: 3.16<br>yaw: -18.05<br>beauty: 53.95<br>expression: none<br>face_prob: 1<br>face_shape: square<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 30<br>pitch: 5.91<br>roll: -1.93<br>yaw: -16.15<br>beauty: 68.01<br>expression: smile<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: male<br>glasses: none<br>race: white|age: 31<br>pitch: 4.87<br>roll: -1.72<br>yaw: -15.08<br>beauty: 78.04<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 25<br>pitch: 3.0<br>roll: 25.22<br>yaw: -39.2<br>beauty: 61.25<br>expression: smile<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: female<br>glasses: none<br>race: white|age: 22<br>pitch: -1.51<br>roll: 24.62<br>yaw: -44.19<br>beauty: 59.63<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 27<br>pitch: -6.54<br>roll: -5.49<br>yaw: -10.31<br>beauty: 32.62<br>expression: none<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: female<br>glasses: none<br>race: white|age: 28<br>pitch: -7.45<br>roll: -8.79<br>yaw: -2.87<br>beauty: 28.23<br>expression: none<br>face_prob: 0.93<br>face_shape: triangle<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 38<br>pitch: 3.89<br>roll: 2.17<br>yaw: 26.96<br>beauty: 24.02<br>expression: none<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: male<br>glasses: none<br>race: white|age: 35<br>pitch: 6.15<br>roll: -0.95<br>yaw: 18.02<br>beauty: 21.73<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: black|
