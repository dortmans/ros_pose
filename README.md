# ros_pose


Install prerequisites:
```
sudo apt install ros-kinetic-cv-camera
```

Usage:

Launch webcam node:
```
roslaunch ros_pose cv_camera.launch
```

Launch pose estimator:
```
roslaunch ros_pose pose_estimator.launch

```

Launch person detector:
```
roslaunch ros_pose person_detector.launch image_topic:=<raw image topic>
```

