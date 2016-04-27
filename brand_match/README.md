README - BRAND MATCH descriptors 

Library Dependencies
----------------------------------------

The BRAND matching demo code is based on OpenCV and PCL libraries.

Getting Started
----------------------------------------

Compiling:

```
mkdir build
cd build
cmake ..
make
``

Running:
``
./brand_match_demo ../../test_images/real-poseA-t01-rgb.png ../../test_images/real-poseA-t01-depth.png ../../test_images/real-poseA-t02-rgb.png ../../test_images/real-poseA-t02-depth.png  525 525 319 239
``

Where, fx = fy = 525, cx = 319, cy = 239 (i.e. the kinect intrinsic parameters)
