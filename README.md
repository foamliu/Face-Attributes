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
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 21<br>pitch: 9.08<br>roll: -3.51<br>yaw: 33.62<br>beauty: 37.2<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 24<br>pitch: 9.28<br>roll: -0.51<br>yaw: 23.2<br>beauty: 66.23<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 26<br>pitch: 4.15<br>roll: -4.63<br>yaw: -2.1<br>beauty: 34.72<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 43<br>pitch: -5.58<br>roll: 7.24<br>yaw: -7.08<br>beauty: 29.92<br>expression: smile<br>face_prob: 1<br>face_shape: triangle<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 25<br>pitch: 12.49<br>roll: -16.73<br>yaw: 25.73<br>beauty: 27.07<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 32<br>pitch: 7.9<br>roll: 17.66<br>yaw: -31.34<br>beauty: 42.21<br>expression: none<br>face_prob: 0.98<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 29<br>pitch: 6.93<br>roll: -5.44<br>yaw: 3.15<br>beauty: 36.7<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 3.92<br>roll: 5.02<br>yaw: -22.21<br>beauty: 43.78<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 29<br>pitch: 6.06<br>roll: 7.76<br>yaw: -14.08<br>beauty: 37.69<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 34<br>pitch: -3.7<br>roll: 12.57<br>yaw: -19.83<br>beauty: 47.4<br>expression: none<br>face_prob: 1<br>face_shape: square<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 26<br>pitch: 6.06<br>roll: -1.91<br>yaw: -4.75<br>beauty: 18.96<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 24<br>pitch: 23.78<br>roll: 18.92<br>yaw: -14.62<br>beauty: 36.02<br>expression: none<br>face_prob: 0.92<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 21<br>pitch: -1.63<br>roll: 7.43<br>yaw: -6.64<br>beauty: 40.31<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 35<br>pitch: 11.73<br>roll: -0.44<br>yaw: 3.55<br>beauty: 48.67<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 36<br>pitch: 17.58<br>roll: 0.25<br>yaw: -5.12<br>beauty: 40.05<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 43<br>pitch: 17.99<br>roll: -6.83<br>yaw: -1.76<br>beauty: 43.23<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: sun<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 38<br>pitch: 12.55<br>roll: 4.33<br>yaw: -18.68<br>beauty: 37.37<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 42<br>pitch: 15.64<br>roll: 4.66<br>yaw: -6.16<br>beauty: 44.98<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 19<br>pitch: 9.46<br>roll: -5.87<br>yaw: -14.09<br>beauty: 46.86<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 31<br>pitch: 18.44<br>roll: -26.35<br>yaw: 35.01<br>beauty: 64.36<br>expression: smile<br>face_prob: 0.98<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
