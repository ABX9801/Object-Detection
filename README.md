# Object Detection

- I recorded a video while playing GTA5 and tried to implement object detection on the video using YOLO-V2.



**The raw video file recorded from gta5**



![Object Detection Demo](dataset/gta52.gif)




**Requirements :**
- python 3.7
- tensorflow 
- opencv
- numpy

**Download the darkflow Repo from [here](https://github.com/thtrieu/darkflow)**
- the code written  in the darkflow repo is in tensorflow version1.
- we can also migrate our tensorflow1 code to tensorflow2 using:
```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
```


**Build the Library**
- open an cmd window and type:
  - python setup.py build_ext --inplace
  or
  - pip install -e 

**Download weights file from [here](https://pjreddie.com/darknet/yolov2/)**


**Demo conversion**
- move the video file into the darkflow-master
- from there, open a cmd window
- use the command: 
    - python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo videofile.mp4 --gpu 1.0 --saveVideo

- videofile.mp4 is the name of your video.
- NOTE: if you do not have the GPU version of tensorflow, leave off the --gpu 1.0
- --saveVideo indicates to save a name video file, which has the boxes around objects

**I suggest the conversion via python using opencv**
- the code is written in main.py.
- if using cpu version we can remove 'gpu': 1.0 from options

**The processed video file**


![Object Detection Demo](output/output1.gif)
