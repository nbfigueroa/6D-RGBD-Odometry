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

###brand_based_odometry

To use this example you will need to install PCL library (www.pointclouds.org) and OpenCV (www.opencv.org). 

Compile Code:
```
cmake ..
```

Run Code:
```
./6D-rgbd-odometry ...
```

#####References:

Nadia Figueroa, Haiwei Dong, and Abdulmotaleb El Saddik. 2015. A Combined Approach Toward Consistent Reconstructions of Indoor Spaces Based on 6D RGB-D Odometry and KinectFusion. ACM Trans. Intell. Syst. Technol. 6, 2, Article 14 (March 2015), 10 pages.

H. Dong, N. Figueroa and A. El Saddik, "Towards consistent reconstructions of indoor spaces based on 6D RGB-D odometry and KinectFusion," 2014 IEEE/RSJ International Conference on Intelligent Robots and Systems, Chicago, IL, 2014, pp. 1796-1803.

---

###brand_match
We also provide a stand-alone example that shows how to use the class to create BRAND descriptors.

Compile Code:
```
cmake ..
```

Run Code:
```
./6D-rgbd-odometry ...
```

#####Reference:

E. R. Nascimento, G. L. Oliveira, M. F. M. Campos, A. W. Vieira and W. R. Schwartz, "BRAND: A robust appearance and depth descriptor for RGB-D images," 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, Vilamoura, 2012, pp. 1720-1726.

Brand Code provided by: [Erickson R. Nascimento](http://homepages.dcc.ufmg.br/~erickson/index.html)

