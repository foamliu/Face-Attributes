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
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 25<br>pitch: 4.26<br>roll: 2.67<br>yaw: -5.94<br>beauty: 30.74<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 34<br>pitch: 9.26<br>roll: 3.14<br>yaw: -10.21<br>beauty: 68.45<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 23<br>pitch: 9.76<br>roll: -3.52<br>yaw: -10.65<br>beauty: 25.31<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 14.63<br>roll: -16.92<br>yaw: 24.37<br>beauty: 33.47<br>expression: none<br>face_prob: 0.96<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: yellow|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 26<br>pitch: 6.0<br>roll: -0.56<br>yaw: -9.86<br>beauty: 27.74<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 8.24<br>roll: -3.46<br>yaw: -26.63<br>beauty: 61.76<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 23<br>pitch: 5.8<br>roll: 28.4<br>yaw: -29.21<br>beauty: 28.58<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 36<br>pitch: 15.73<br>roll: -35.82<br>yaw: 4.97<br>beauty: 67.17<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 22<br>pitch: 7.68<br>roll: 2.43<br>yaw: -8.14<br>beauty: 25.62<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 33<br>pitch: 28.45<br>roll: 3.68<br>yaw: 9.04<br>beauty: 42.43<br>expression: none<br>face_prob: 0.98<br>face_shape: heart<br>face_type: human<br>gender: female<br>glasses: none<br>race: black|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 25<br>pitch: 11.26<br>roll: -9.41<br>yaw: -9.41<br>beauty: 21.88<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 40<br>pitch: 8.33<br>roll: -10.61<br>yaw: 13.57<br>beauty: 39.2<br>expression: none<br>face_prob: 0.97<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: common<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 25<br>pitch: 7.57<br>roll: -4.97<br>yaw: 13.25<br>beauty: 26.17<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 32<br>pitch: 7.79<br>roll: -9<br>yaw: -21.81<br>beauty: 25.46<br>expression: none<br>face_prob: 0.68<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 25<br>pitch: 4.91<br>roll: -4.63<br>yaw: 11.24<br>beauty: 26.84<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 9.5<br>roll: 9.23<br>yaw: -5.2<br>beauty: 60.27<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 26<br>pitch: 11.46<br>roll: -5.48<br>yaw: -11.89<br>beauty: 24.5<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 50<br>pitch: 8.6<br>roll: 1.59<br>yaw: -4.64<br>beauty: 55.29<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 22<br>pitch: 6.27<br>roll: -3.64<br>yaw: -16.56<br>beauty: 28.56<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 31<br>pitch: 25.64<br>roll: -10.93<br>yaw: 38.87<br>beauty: 54.96<br>expression: none<br>face_prob: 0.95<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
