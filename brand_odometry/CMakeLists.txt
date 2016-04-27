cmake_minimum_required (VERSION 2.6)
project (BRAND_ODOMETRY)

find_package (OpenCV REQUIRED) 
find_package (PCL 1.4 REQUIRED)

include_directories (include ${PCL_INCLUDE_DIRS} ${BRAND_INCLUDE_DIRS})

add_executable ( 6d_odometry_nadia src/brand_match_demo.cpp )
target_link_libraries ( 6d_odometry_nadia
   pcl_features
   pcl_common
   pcl_search
   pcl_kdtree
   ${OpenCV_LIBS} brand)

