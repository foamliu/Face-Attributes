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

Image | Aligned | Out | True | Angles |
|---|---|---|---|---|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 39<br>pitch: 9.55<br>roll: 2.92<br>yaw: -4.37<br>beauty: 60.22<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 38<br>pitch: 14.47<br>roll: 4.72<br>yaw: -6.27<br>beauty: 61.63<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 34<br>pitch: -0.62<br>roll: -3.33<br>yaw: -13.79<br>beauty: 56.74<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 33<br>pitch: 0.53<br>roll: -5.75<br>yaw: -15.31<br>beauty: 50.99<br>expression: none<br>face_prob: 1<br>face_shape: square<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 28<br>pitch: 7.97<br>roll: 30.63<br>yaw: -14.23<br>beauty: 70.43<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 35<br>pitch: 11.85<br>roll: 29.4<br>yaw: -9.4<br>beauty: 63.73<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 26<br>pitch: 18.03<br>roll: -8.83<br>yaw: 5.02<br>beauty: 49.93<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 18<br>pitch: 24.07<br>roll: -9.4<br>yaw: 8.33<br>beauty: 39.33<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: sun<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 39<br>pitch: 5.49<br>roll: -18.05<br>yaw: 1.98<br>beauty: 47.5<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 40<br>pitch: 9.13<br>roll: -17.92<br>yaw: 3.36<br>beauty: 60.21<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: black|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 42<br>pitch: 2.94<br>roll: 4.65<br>yaw: -20.59<br>beauty: 43.61<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 44<br>pitch: 4.78<br>roll: 5.68<br>yaw: -22.54<br>beauty: 41.06<br>expression: none<br>face_prob: 0.8<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 44<br>pitch: 17.16<br>roll: -23.33<br>yaw: 8.27<br>beauty: 41.2<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 48<br>pitch: 19.35<br>roll: -23.83<br>yaw: 12.14<br>beauty: 51.83<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 32<br>pitch: 10.4<br>roll: -7.64<br>yaw: -0.19<br>beauty: 62.59<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 12.35<br>roll: -6.42<br>yaw: 6.47<br>beauty: 55.07<br>expression: laugh<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: common<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 34<br>pitch: 10.17<br>roll: 9.78<br>yaw: 1.05<br>beauty: 73.98<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 32<br>pitch: 10.55<br>roll: 7.34<br>yaw: 5.17<br>beauty: 79.6<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_angle.jpg)|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 30<br>pitch: 3.94<br>roll: 4.44<br>yaw: -2.74<br>beauty: 49.85<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 28<br>pitch: 7<br>roll: 4.96<br>yaw: -3.78<br>beauty: 47<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_angle.jpg)|
