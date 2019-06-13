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
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 32<br>pitch: 4.07<br>roll: 3.66<br>yaw: 21.74<br>beauty: 21.73<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 32<br>pitch: -0.63<br>roll: -2.11<br>yaw: 21.2<br>beauty: 11.68<br>expression: none<br>face_prob: 0.82<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: black|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 38<br>pitch: 13.58<br>roll: -4.14<br>yaw: 2.05<br>beauty: 58.47<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 40<br>pitch: 14.41<br>roll: -5.7<br>yaw: -0.05<br>beauty: 51.77<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 33<br>pitch: 3.3<br>roll: 1.22<br>yaw: -13.07<br>beauty: 42.56<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 42<br>pitch: 4.14<br>roll: 5.94<br>yaw: -9.98<br>beauty: 38.6<br>expression: none<br>face_prob: 0.95<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 35<br>pitch: -3.6<br>roll: -7.65<br>yaw: 4.84<br>beauty: 52.85<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 35<br>pitch: -11.95<br>roll: -5.22<br>yaw: 2.25<br>beauty: 51.55<br>expression: none<br>face_prob: 0.98<br>face_shape: square<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 32<br>pitch: 19.51<br>roll: -18.56<br>yaw: 27.52<br>beauty: 50.41<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 31<br>pitch: 23.86<br>roll: -21.49<br>yaw: 25.66<br>beauty: 46.15<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 27<br>pitch: 13.44<br>roll: -11.89<br>yaw: -9.66<br>beauty: 57.96<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 27<br>pitch: 13.96<br>roll: -11.73<br>yaw: -8.06<br>beauty: 49.86<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 31<br>pitch: 12.34<br>roll: -21.99<br>yaw: 14.79<br>beauty: 44.1<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 36<br>pitch: 13.32<br>roll: -21.89<br>yaw: 13.5<br>beauty: 49.8<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 26<br>pitch: 7.45<br>roll: -3.84<br>yaw: 28.75<br>beauty: 41.28<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 28<br>pitch: 12.16<br>roll: -4.67<br>yaw: 30.19<br>beauty: 45.31<br>expression: smile<br>face_prob: 0.88<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 28<br>pitch: 12.69<br>roll: -12.89<br>yaw: 60.56<br>beauty: 36.2<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 31<br>pitch: 13.25<br>roll: -13.65<br>yaw: 56.77<br>beauty: 37.66<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 30<br>pitch: 7.05<br>roll: -2.26<br>yaw: 3.55<br>beauty: 63.83<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 32<br>pitch: 8.02<br>roll: 0.39<br>yaw: 2.64<br>beauty: 73.06<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
