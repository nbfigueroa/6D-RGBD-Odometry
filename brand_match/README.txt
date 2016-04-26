README - BRAND descriptors 

Library Dependencies
----------------------------------------

The BRAND code is based on the OpenCV library and the demo code is based on OpenCV and PCL libraries.

Getting Started
----------------------------------------

Compiling:

mkdir build
cd build
cmake ..
make

Running:

./brand_match_demo rgb1.png depth1.png rgb2.png depth2.png 525 525 319 239

Where, fx = fy = 525, cx = 319, cy = 239
