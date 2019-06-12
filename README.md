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
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 26<br>pitch: 4.52<br>roll: -2.82<br>yaw: 22.56<br>beauty: 29.07<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 34<br>pitch: -0.99<br>roll: -4.68<br>yaw: 53.07<br>beauty: 28.63<br>expression: none<br>face_prob: 0.99<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: yellow|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 29<br>pitch: 6.96<br>roll: 2.37<br>yaw: -20.3<br>beauty: 26.63<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 66<br>pitch: 15.17<br>roll: -12.52<br>yaw: 10.39<br>beauty: 40.51<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 26<br>pitch: 10.3<br>roll: -8.48<br>yaw: -12.15<br>beauty: 21.45<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 42<br>pitch: 15.02<br>roll: -4.44<br>yaw: 33.2<br>beauty: 32.17<br>expression: smile<br>face_prob: 0.99<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 25<br>pitch: 8.5<br>roll: -3.14<br>yaw: 4.93<br>beauty: 27.0<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 25<br>pitch: 4.56<br>roll: -0.43<br>yaw: 6.01<br>beauty: 57.32<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 25<br>pitch: 6.3<br>roll: -3.95<br>yaw: 2.78<br>beauty: 28.92<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 8.28<br>roll: 3.67<br>yaw: -63.99<br>beauty: 30.46<br>expression: none<br>face_prob: 0.99<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 24<br>pitch: 9.18<br>roll: -3.81<br>yaw: -13.91<br>beauty: 27.95<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 31<br>pitch: 7.88<br>roll: 1.32<br>yaw: -16.95<br>beauty: 52.67<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 24<br>pitch: 5.63<br>roll: -3.44<br>yaw: 3.21<br>beauty: 25.2<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 33<br>pitch: 9.38<br>roll: -4.44<br>yaw: 1.03<br>beauty: 54.73<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 25<br>pitch: 8.32<br>roll: -5.67<br>yaw: -7.31<br>beauty: 26.19<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 13.93<br>roll: 2.48<br>yaw: 80.14<br>beauty: 33.81<br>expression: none<br>face_prob: 0.86<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 24<br>pitch: 5.45<br>roll: -3.18<br>yaw: -10.2<br>beauty: 31.68<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 23<br>pitch: 8.24<br>roll: -8.6<br>yaw: 69.1<br>beauty: 38.02<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: yellow|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 31<br>pitch: 10.79<br>roll: -15.01<br>yaw: 14.04<br>beauty: 20.12<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 44<br>pitch: 7.46<br>roll: -2.98<br>yaw: -12.51<br>beauty: 59.69<br>expression: smile<br>face_prob: 0.99<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
