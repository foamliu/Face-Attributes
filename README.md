# Face-Attributes

Deep Face Attributes.


## DataSet

CASIA WebFace DataSet, 479,653 faces.

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
