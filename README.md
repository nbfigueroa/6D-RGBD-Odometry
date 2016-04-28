## 6D-RGBD-Odometry

This package provides a novel  6D  RGB-D   odometry   approach   that   finds   the   relative   camera   pose between consecutive RGB-D frames by keypoint extraction and feature  matching  both  on  the  RGB  and  depth  image  planes.

To find the Rigid Motion between two camera frames, the VO algorithm follows these steps:

#####1) RGB-D Keypoint Extraction:

STAR Detector for RGB image and NARF detector for Depth Image.

#####2) RGB-D Feature Matching:

BRAND Description computation on keypoints and brute-force descriptor matcher with the Hamming norm and correspondence cross checking.

#####3) Correspondences Outlier Rejection:

Filter the matches using RANSAC outlier rejection algorithm  

#####4) 6D Rigid Motion Estimation:

Iterative estimation using RANSAC and Umeyama Method.

The output of this algorithm should be something similar to this video:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=zR4nMKZL8go
" target="_blank"><img src="http://img.youtube.com/vi/zR4nMKZL8go/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

---

To use this code, you will need to install PCL library (www.pointclouds.org), OpenCV (www.opencv.org) and compile the brand library in this folder, to do so,  (after succesfully installing PCL and OpenCV) you must follow these instructions:

```
cd ~/brand/
mkdir build && cd build
make
sudo make install
```
Once compiled you can test the BRAND desciptors by running a simple matching example found in:

###brand_match
This stand-alone example shows how to use the class to create BRAND descriptors and a simple brute-force matching between two frames.

Compiling:
```
cd ~/brand_match/
mkdir build && cd build
make
```
Running
```
./brand_match_demo ../../test_images/real-poseA-t01-rgb.png ../../test_images/real-poseA-t01-depth.png ../../test_images/real-poseA-t02-rgb.png ../../test_images/real-poseA-t02-depth.png  525 525 319 239
```
Where, fx = fy = 525, cx = 319, cy = 239 (i.e. the kinect intrinsic parameters)

You should then get the following output:

![alt text](https://github.com/nbfigueroa/6D-RGBD-Odometry/blob/master/brand_match/brand_matches.png "BRAND Descriptot Matches")


###brand_odometry
Finally, if you want to test the odometry algorithm, you should download a dataset with a moving rgb-d sensor. I recommend one of these : [TUM RGB-D Dataset](http://vision.in.tum.de/data/datasets/rgbd-dataset/download) or if you have your own dataset you should just put both rgb and depth in the same folder with same timestamps, follow the naming convention of the rgb-d dataset. 

** Not working, missing a header file **

Compiling:
```
cd ~/brand_odometry/
mkdir build && cd build
make
```
Running:
```
./brand_odometry ${path-to-dataset} 
```

---
#####Reference:
Nadia Figueroa, Haiwei Dong, and Abdulmotaleb El Saddik. 2015. A Combined Approach Toward Consistent Reconstructions of Indoor Spaces Based on 6D RGB-D Odometry and KinectFusion. ACM Trans. Intell. Syst. Technol. 6, 2, Article 14 (March 2015), 10 pages.

H. Dong, N. Figueroa and A. El Saddik, "Towards consistent reconstructions of indoor spaces based on 6D RGB-D odometry and KinectFusion," 2014 IEEE/RSJ International Conference on Intelligent Robots and Systems, Chicago, IL, 2014, pp. 1796-1803.

E. R. Nascimento, G. L. Oliveira, M. F. M. Campos, A. W. Vieira and W. R. Schwartz, "BRAND: A robust appearance and depth descriptor for RGB-D images," 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, Vilamoura, 2012, pp. 1720-1726.

Brand Code provided by: [Erickson R. Nascimento](http://homepages.dcc.ufmg.br/~erickson/index.html)
6D-RGD-Odometry Code provided by Nadia Figueroa.
