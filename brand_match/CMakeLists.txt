cmake_minimum_required (VERSION 2.6)
project (BRAND_DESCRIPTOR_DEMO)

find_package (OpenCV REQUIRED) 
find_package (PCL 1.4 REQUIRED)

include_directories (include ${PCL_INCLUDE_DIRS} ${BRAND_INCLUDE_DIRS})

add_executable ( brand_match_demo src/brand_match_demo.cpp )
target_link_libraries ( brand_match_demo
   pcl_features
   pcl_common
   pcl_search
   pcl_kdtree
   ${OpenCV_LIBS} brand)
