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
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 30<br>pitch: 8.77<br>roll: -8.59<br>yaw: -51.2<br>beauty: 31.85<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 31<br>pitch: 7.43<br>roll: 18.52<br>yaw: -52.35<br>beauty: 25.22<br>expression: laugh<br>face_prob: 0.93<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: yellow|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 28<br>pitch: 15.4<br>roll: -17.36<br>yaw: -82.07<br>beauty: 31.12<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 9.92<br>roll: 22.75<br>yaw: -75.85<br>beauty: 20.68<br>expression: none<br>face_prob: 0.92<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 37<br>pitch: 6.7<br>roll: 0.77<br>yaw: -23.95<br>beauty: 64.75<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 40<br>pitch: 9.14<br>roll: 4.48<br>yaw: -29.19<br>beauty: 60.94<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 41<br>pitch: 5.06<br>roll: 2.65<br>yaw: -8.47<br>beauty: 55.87<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 46<br>pitch: 6.83<br>roll: 0.62<br>yaw: -10.98<br>beauty: 49.34<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 33<br>pitch: 10.41<br>roll: -5.1<br>yaw: -5.9<br>beauty: 55.31<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 29<br>pitch: 12.31<br>roll: 4.46<br>yaw: -3.41<br>beauty: 50.52<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: yellow|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 24<br>pitch: 4.89<br>roll: -5.77<br>yaw: 77.28<br>beauty: 38.68<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 25<br>pitch: 9.85<br>roll: -22.82<br>yaw: 84.83<br>beauty: 38.36<br>expression: none<br>face_prob: 0.87<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 52<br>pitch: 16.12<br>roll: -12.39<br>yaw: 11.05<br>beauty: 45.42<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 57<br>pitch: 14.33<br>roll: -15.39<br>yaw: 8.61<br>beauty: 40.01<br>expression: smile<br>face_prob: 0.99<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: common<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 38<br>pitch: 1.8<br>roll: -1.93<br>yaw: -10.67<br>beauty: 55.55<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 37<br>pitch: 1.56<br>roll: -3.44<br>yaw: -14.52<br>beauty: 46.95<br>expression: none<br>face_prob: 1<br>face_shape: square<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 32<br>pitch: 2.63<br>roll: 8.23<br>yaw: 15.68<br>beauty: 55.93<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 32<br>pitch: 2.28<br>roll: -26.2<br>yaw: 12.17<br>beauty: 45.8<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 33<br>pitch: 5.49<br>roll: 0.16<br>yaw: -36.65<br>beauty: 58.3<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 31<br>pitch: 9.01<br>roll: -3.5<br>yaw: -35.1<br>beauty: 62.83<br>expression: none<br>face_prob: 0.96<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
