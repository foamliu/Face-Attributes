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
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/0_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/0_img.jpg)|age: 32<br>pitch: 8.26<br>roll: -4.09<br>yaw: -0.95<br>beauty: 48.75<br>expression: none<br>face_prob: 0.7677<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|age: 36<br>pitch: 8.13<br>roll: -8.54<br>yaw: 42.44<br>beauty: 47.49<br>expression: none<br>face_prob: 0.86<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: common<br>race: white|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/1_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/1_img.jpg)|age: 32<br>pitch: 8.07<br>roll: -3.99<br>yaw: -1.41<br>beauty: 48.75<br>expression: none<br>face_prob: 0.7648<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|age: 35<br>pitch: 7.83<br>roll: 6.93<br>yaw: 17.39<br>beauty: 40.61<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: common<br>race: white|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/2_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/2_img.jpg)|age: 32<br>pitch: 8.32<br>roll: -4.07<br>yaw: -0.82<br>beauty: 48.75<br>expression: none<br>face_prob: 0.7687<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|age: 37<br>pitch: 1.89<br>roll: 4.37<br>yaw: 21.98<br>beauty: 29.69<br>expression: none<br>face_prob: 0.95<br>face_shape: round<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/3_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/3_img.jpg)|age: 32<br>pitch: 8.32<br>roll: -4.09<br>yaw: -0.81<br>beauty: 48.75<br>expression: none<br>face_prob: 0.7686<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|age: 23<br>pitch: 10.95<br>roll: -14.12<br>yaw: -3.75<br>beauty: 48<br>expression: none<br>face_prob: 0.87<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: black|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/4_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/4_img.jpg)|age: 32<br>pitch: 8.09<br>roll: -4.11<br>yaw: -1.41<br>beauty: 48.76<br>expression: none<br>face_prob: 0.7648<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|age: 28<br>pitch: 21.48<br>roll: -2.47<br>yaw: -3.53<br>beauty: 42.99<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/5_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/5_img.jpg)|age: 32<br>pitch: 8.32<br>roll: -4.09<br>yaw: -0.8<br>beauty: 48.75<br>expression: none<br>face_prob: 0.7687<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|age: 23<br>pitch: 8.42<br>roll: -17.64<br>yaw: 77.12<br>beauty: 31.74<br>expression: none<br>face_prob: 0.96<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/6_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/6_img.jpg)|age: 32<br>pitch: 8.16<br>roll: -4.07<br>yaw: -1.21<br>beauty: 48.75<br>expression: none<br>face_prob: 0.7661<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|age: 31<br>pitch: 23.92<br>roll: -15.42<br>yaw: 19.38<br>beauty: 33.34<br>expression: none<br>face_prob: 0.95<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/7_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/7_img.jpg)|age: 32<br>pitch: 7.47<br>roll: -4.43<br>yaw: -2.88<br>beauty: 48.76<br>expression: smile<br>face_prob: 0.7558<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|age: 35<br>pitch: 18.75<br>roll: -3.63<br>yaw: -0.26<br>beauty: 58.88<br>expression: smile<br>face_prob: 0.98<br>face_shape: oval<br>face_type: human<br>gender: female<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/8_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/8_img.jpg)|age: 32<br>pitch: 7.98<br>roll: -4.3<br>yaw: -1.7<br>beauty: 48.76<br>expression: none<br>face_prob: 0.7633<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|age: 25<br>pitch: 20.34<br>roll: -2.22<br>yaw: -6.85<br>beauty: 64.57<br>expression: smile<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|
|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/9_raw.jpg)|![image](https://github.com/foamliu/Face_Attributes/raw/master/images/9_img.jpg)|age: 32<br>pitch: 8.27<br>roll: -4.08<br>yaw: -0.93<br>beauty: 48.75<br>expression: none<br>face_prob: 0.7679<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|age: 36<br>pitch: 21.82<br>roll: 7.21<br>yaw: 10.6<br>beauty: 68.71<br>expression: none<br>face_prob: 1<br>face_shape: oval<br>face_type: human<br>gender: male<br>glasses: none<br>race: white|