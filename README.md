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
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/0_img.jpg)|age: 28<br>pitch: 7.98<br>roll: -1.45<br>yaw: -5.08<br>beauty: 37.75<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 42<br>pitch: 8.34<br>roll: 9.48<br>yaw: -5.7<br>beauty: 55.11<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/1_img.jpg)|age: 28<br>pitch: 11.52<br>roll: -4.65<br>yaw: 5.58<br>beauty: 35.33<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 31<br>pitch: 7.75<br>roll: 0.05<br>yaw: -5.29<br>beauty: 67.02<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/2_img.jpg)|age: 23<br>pitch: 8.1<br>roll: -4.16<br>yaw: 2.97<br>beauty: 30.28<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 27<br>pitch: 9.47<br>roll: -17.92<br>yaw: 6.63<br>beauty: 50.16<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: yellow|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/3_img.jpg)|age: 24<br>pitch: 14.92<br>roll: 4.02<br>yaw: -51.0<br>beauty: 19.25<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 29<br>pitch: 25.25<br>roll: -23.34<br>yaw: 53.89<br>beauty: 40.64<br>expression: none<br>face_prob: 0.97<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/4_img.jpg)|age: 25<br>pitch: 18.1<br>roll: -2.63<br>yaw: -1.34<br>beauty: 44.24<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 27<br>pitch: 16.95<br>roll: 7.85<br>yaw: -4.41<br>beauty: 68.98<br>expression: none<br>face_prob: 0.91<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/5_img.jpg)|age: 27<br>pitch: 6.77<br>roll: -5.13<br>yaw: -11.39<br>beauty: 30.96<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 26<br>pitch: 8.14<br>roll: -12.98<br>yaw: 11.5<br>beauty: 46.56<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/6_img.jpg)|age: 38<br>pitch: 9.38<br>roll: 0.65<br>yaw: -15.64<br>beauty: 28.88<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 42<br>pitch: 5.26<br>roll: -7.63<br>yaw: 25.72<br>beauty: 42.45<br>expression: smile<br>face_prob: 1<br>face_shape: round<br>face_type: human<br>gender: male<br>glasses: none<br>race: black|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/7_img.jpg)|age: 39<br>pitch: 2.73<br>roll: -4.48<br>yaw: -3.16<br>beauty: 30.28<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 40<br>pitch: -1.18<br>roll: -7.15<br>yaw: 7.3<br>beauty: 24.02<br>expression: smile<br>face_prob: 1<br>face_shape: square<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/8_img.jpg)|age: 28<br>pitch: 19.41<br>roll: 3.91<br>yaw: -26.62<br>beauty: 36.87<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 30<br>pitch: 27.14<br>roll: -7.82<br>yaw: 27.71<br>beauty: 59.6<br>expression: smile<br>face_prob: 0.98<br>face_shape: heart<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face-Attributes/raw/master/images/9_img.jpg)|age: 22<br>pitch: 11.44<br>roll: -3.72<br>yaw: 4.48<br>beauty: 30.4<br>expression: None<br>face_prob: None<br>face_shape: None<br>face_type: None<br>gender: None<br>glasses: None<br>race: None|age: 23<br>pitch: 28.83<br>roll: -20.64<br>yaw: 10.43<br>beauty: 31.78<br>expression: none<br>face_prob: 0.96<br>face_shape: round<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
