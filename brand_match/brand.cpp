/* 
   Copyright (C) 2012-2013 Erickson R. Nascimento

   THIS SOURCE CODE IS PROVIDED 'AS-IS', WITHOUT ANY EXPRESS OR IMPLIED
   WARRANTY. IN NO EVENT WILL THE AUTHOR BE HELD LIABLE FOR ANY DAMAGES
   ARISING FROM THE USE OF THIS SOFTWARE.

   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:


   1. The origin of this source code must not be misrepresented; you must not
      claim that you wrote the original source code. If you use this source code
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original source code.

   3. This notice may not be removed or altered from any source distribution.

   Contact: erickson [at] dcc [dot] ufmg [dot] br

*/

#include "brand.h"

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/legacy/compat.hpp>
#include <opencv2/nonfree/nonfree.hpp> 
#include <opencv2/contrib/contrib.hpp>

static const float DEGREE2RAD = (float)CV_PI/180.0;

// degree_threshold in degree
// descriptor_size  in bytes
BrandDescriptorExtractor::BrandDescriptorExtractor( double degree_threshold, int descriptor_size ) 
{
   m_descriptor_size = descriptor_size;
   m_degree_threshold  = cos( degree_threshold*DEGREE2RAD );

   cv::initModule_nonfree(); // to use SURF canonical estimation
}

void BrandDescriptorExtractor::compute(const cv::Mat& image, const cv::Mat& cloud, 
                                       const cv::Mat& normals,
                                       std::vector<cv::KeyPoint>& keypoints,
                                       cv::Mat& descriptors )
{
   cv::Mat intensity_descriptors, shape_descriptors;
   
   extract_features( cloud, normals, image, keypoints, intensity_descriptors, shape_descriptors );
   bitwise_or(intensity_descriptors, shape_descriptors, descriptors);
}

void BrandDescriptorExtractor::canonical_orientation(  const cv::Mat& img, const cv::Mat& mask,
                                                       std::vector<cv::KeyPoint>& keypoints ) 
{
    cv::Ptr<cv::Feature2D> surf = cv::Algorithm::create<cv::Feature2D>("Feature2D.SURF");
    if( surf.empty() )
        CV_Error(CV_StsNotImplemented, "OpenCV was built without SURF support.");

    surf->operator()(img, cv::noArray(), keypoints, cv::noArray(), true);
}


void BrandDescriptorExtractor::extract_features(const cv::Mat& cloud, const cv::Mat& normals,
                                                const cv::Mat &image, std::vector<cv::KeyPoint>& keypoints, 
                                                cv::Mat& intensity, cv::Mat& shape )
{
   for(int i = 0; i < keypoints.size(); ++i) 
   {
      double depth = cloud.at<cv::Point3f>(keypoints[i].pt.y, keypoints[i].pt.x).z;
      // scale pairs of pixel distribution
      keypoints[i].response = std::max( 0.2, (3.8-0.4*std::max(2.0, depth))/3); 
      // used to define the size of HAAR wavelets
      keypoints[i].size = 70.0 * keypoints[i].response;    
   }

   canonical_orientation( image, cv::Mat(), keypoints );
   compute_intensity_and_shape_descriptors( image, cloud, normals, keypoints, intensity, shape );
}

void BrandDescriptorExtractor::compute_intensity_and_shape_descriptors( const cv::Mat& image,
                                                                        const cv::Mat& cloud,
                                                                        const cv::Mat& normals,
                                                                        std::vector<cv::KeyPoint>& keypoints,
                                                                        cv::Mat& idescriptors,
                                                                        cv::Mat& sdescriptors )
{
    // Construct integral image for fast smoothing (box filter)
    cv::Mat sum;

    cv::Mat grayImage = image;
    if( image.type() != CV_8U ) cvtColor( image, grayImage, CV_BGR2GRAY );

    integral( grayImage, sum, CV_32S);

    //Remove keypoints very close to the border
    cv::KeyPointsFilter::runByImageBorder( keypoints, image.size(), m_patch_size + m_half_kernel_size );

    idescriptors = cv::Mat::zeros((int)keypoints.size(), m_descriptor_size, CV_8U);
    sdescriptors = cv::Mat::zeros((int)keypoints.size(), m_descriptor_size, CV_8U);
    pixelTests(sum, cloud, normals, keypoints, idescriptors, sdescriptors);
}


inline int BrandDescriptorExtractor::smoothedSum(const cv::Mat& sum, const cv::KeyPoint& kpt, cv::Point2f& pt)
{
    pt.x += (int)(kpt.pt.x + 0.5);
    pt.y += (int)(kpt.pt.y + 0.5);

    return   sum.at<int>(pt.y + m_half_kernel_size + 1, pt.x + m_half_kernel_size + 1)
           - sum.at<int>(pt.y + m_half_kernel_size + 1, pt.x - m_half_kernel_size)
           - sum.at<int>(pt.y - m_half_kernel_size,     pt.x + m_half_kernel_size + 1)
           + sum.at<int>(pt.y - m_half_kernel_size,     pt.x - m_half_kernel_size);
}

void BrandDescriptorExtractor::pixelTests(  const cv::Mat& sum,
                                            const cv::Mat& cloud,
                                            const cv::Mat& normals,
                                            const std::vector<cv::KeyPoint>& keypoints, 
                                            cv::Mat& idescriptors, cv::Mat& sdescriptors )
{
   //x1,y1,x2,y2
   int bit_pattern[512 * 4] = { -7,-8,2,-6, -1,5,-8,-20, -15,2,-7,-5,
   14,18,2,5, 4,-13,14,-18, 11,-19,-2,-13, 10,9,-12,-12, -24,0,16,10,
   11,-20,-6,-15, -14,-17,22,8, 6,14,1,-23, 16,-4,10,-9, -4,13,20,0, 8,21,-5,3,
   7,6,3,-23, 9,-21,12,19, -6,-6,-2,0, 6,-18,-4,-11, 16,-12,-1,15, 14,3,-4,16,
   17,-2,-13,-19, -2,-7,-9,-17, 5,5,5,6, 14,2,7,-6, -4,17,16,-5, 23,-1,7,20,
   -5,10,-5,-22, -8,6,-5,6, -13,17,9,-18, -20,3,9,9, -13,16,12,-4, -3,9,13,-14,
   3,-7,3,-5, -12,-9,16,-4, 22,-5,-9,3, 8,-2,18,-7, 10,-19,-7,20, 5,-17,1,-7,
   0,24,-1,-11, -10,-13,-6,1, -9,8,-4,9, -12,-3,0,11, -16,-8,1,2, 0,-9,2,0,
   12,-3,-21,3, -16,-16,-13,20, 6,5,9,13, 11,17,-16,7, 8,5,-14,-7, -12,5,8,-2,
   -8,21,-9,19, -7,-10,-21,-4, 2,-5,-8,5, -16,3,-14,-8, -8,-1,-14,-5,
   -20,-5,-4,13, 14,17,-16,-16, -14,-8,-7,18, 4,15,11,-13, 19,-11,-12,-4,
   2,13,4,12, 13,-8,6,14, -5,19,2,-9, 17,-12,22,7, -18,4,10,-20, -6,15,-4,-3,
   -3,11,-20,-11, 14,9,-9,-7, -9,11,1,18, -6,-20,13,-8, -2,-18,23,4, -21,-5,17,-1,
   -1,-9,10,1, -2,14,-13,-20, 22,6,-13,13, -14,-14,0,-4, 21,-5,1,-8, -21,0,0,13,
   -20,-4,8,-20, -6,0,-13,-9, -13,-16,23,2, 14,-8,7,4, 17,16,14,-17, 16,7,21,-10,
   3,-9,1,10, 1,-22,6,-10, -21,2,12,-17, -18,-8,1,-11, -10,-19,-20,-9, -3,-2,1,8,
   15,17,6,-15, -11,3,-3,0, 15,8,10,9, -17,4,-1,20, 1,19,-1,-6, 2,-22,-9,-19,
   -6,17,23,4, 15,-18,-19,-3, -21,10,20,9, 10,17,-9,-6, -2,2,-2,4, -14,17,11,17,
   -13,-2,-11,5, -13,5,-9,21, 5,2,7,-18, 23,1,-23,-2, -11,8,7,-4, -14,-13,-6,-18,
   18,11,-1,-7, 12,-6,2,8, 19,-14,9,17, -14,0,-4,12, 11,18,-18,9, -20,-13,0,-21,
   14,-15,-9,0, 12,-8,-6,15, 2,14,16,7, -20,-1,16,10, 10,20,-8,-16, 2,-10,4,-15,
   22,-4,16,-12, -18,-12,14,-13, -13,20,14,8, 6,-20,16,5, 20,-11,2,19,
   8,-15,-16,3, 4,12,-7,5, 3,-17,-5,6, 5,-8,17,6, 2,-8,20,12, 0,9,4,12,
   14,-4,-4,12, 22,-2,-7,-6, 19,-6,0,6, -11,-18,15,0, -7,-18,-9,16, 0,-19,7,-22,
   3,12,4,13, 5,-9,1,21, -12,0,19,10, 0,-2,-6,21, 2,7,0,10, -13,9,-14,-17,
   1,11,-19,11, 1,-8,-23,-4, 16,14,15,-16, 9,5,-6,-4, 13,19,22,-8, 1,-12,-19,-2,
   -15,-12,-2,4, 21,8,6,-6, -9,12,5,18, -2,-4,-16,2, 21,-7,-15,10, 4,-16,-11,18,
   22,-8,-4,4, 0,21,-8,-9, 20,2,9,4, -15,-14,12,-11, 6,-13,-18,-12, 23,5,17,-14,
   -15,4,5,-19, -1,-20,-2,20, 10,-17,12,-18, -6,-11,18,4, -11,-17,-1,-15,
   17,9,10,-5, -11,6,16,-7, 2,0,-13,-10, 9,-2,-15,-6, -2,-14,-9,-19, 8,15,-4,17,
   -15,17,20,9, -10,-4,9,-21, 6,-7,-17,-7, 13,7,10,4, 4,-14,-15,11, -20,-3,-9,17,
   2,8,-17,10, -13,-17,-13,-19, -17,-4,-3,21, 7,-17,0,22, -7,-19,23,1,
   -19,3,2,-17, 6,-21,20,-6, -2,15,-6,17, 0,7,-5,14, -20,-12,18,-6, -2,-19,-11,13,
   -21,4,-12,8, 16,-7,-4,-13, -1,15,19,6, 1,12,-14,-5, -20,-7,14,-8,
   -8,-17,-15,-14, -4,-23,15,-18, -14,-1,11,21, -1,0,-19,-8, -23,4,9,2,
   8,-16,-4,17, 5,12,-19,3, 16,8,-14,-7, -2,-1,-7,2, -20,1,9,16, -14,-11,3,22,
   6,21,-14,-1, 8,17,0,-10, -5,8,-14,2, 8,3,21,-10, 11,11,-6,3, -16,17,-8,13,
   21,-3,-8,-8, 14,-11,19,-14, 0,-4,9,20, 17,15,-12,-20, 9,2,-6,11, 0,14,-8,-17,
   10,-14,-20,9, 10,-16,3,-8, 15,6,13,2, 23,4,15,16, -9,-22,-18,-15, -10,-6,19,7,
   -9,21,-1,12, -19,-9,-2,2, -10,-1,14,17, 6,-5,4,17, -11,-8,-1,4, -11,15,-15,10,
   1,-16,-14,8, -12,19,-4,3, -18,2,-2,-7, 21,-6,10,-20, -9,22,0,19, 12,-7,-5,20,
   16,9,-6,10, 3,21,-14,-17, 14,8,-18,-10, 2,21,21,-8, 17,5,6,19, -2,-11,-8,6,
   -23,6,-10,15, -14,7,-2,-21, -12,-19,6,9, 5,15,8,-4, 5,-2,-16,13, 9,-5,-22,5,
   -6,7,-12,-14, -17,2,-8,-11, -4,-11,15,18, 18,-12,12,-12, 14,-2,3,5,
   -15,-6,3,-13, -8,14,-6,15, -11,11,0,-18, 7,-3,-7,19, -8,-10,6,12, 15,-10,-5,5,
   -4,9,16,1, -6,8,15,6, 8,-17,18,3, -17,-14,-14,-13, 3,-16,5,-18, -19,-4,-3,-2,
   18,7,-17,9, 21,0,-9,4, 8,11,15,13, 16,0,-13,3, -11,-12,8,7, 12,-19,3,19,
   -12,17,-12,4, 3,9,-4,-15, -18,-11,8,-1, 5,-16,9,2, 0,-12,-12,2, 5,-4,2,19,
   1,-7,10,-17, -13,2,14,8, 2,-12,4,-19, 8,1,-15,4, 13,12,-20,2, -8,-21,-12,6,
   -2,20,5,-16, -5,-9,-9,4, -8,-3,5,19, -3,0,-2,-10, -18,8,22,-2, -8,1,-13,4,
   14,-1,14,8, 8,15,-12,-10, 13,-19,21,0, 2,-19,13,8, -14,16,-7,-10, 16,5,8,10,
   5,22,-4,-11, 2,9,-10,-5, 8,-14,3,-1, -16,-9,6,-1, 6,-19,-6,-11, -5,23,-17,7,
   -16,-6,-12,5, 12,9,6,21, 23,-5,-6,-12, 8,-9,-1,5, 19,0,-2,-3, 9,-5,13,12,
   11,-15,2,9, -1,6,-11,19, 11,21,-8,-16, -6,4,14,-11, 19,-5,8,7, -23,1,7,7,
   8,-18,-4,-2, 12,15,16,5, -16,-12,-16,11, -8,-21,18,-12, -1,10,-9,-6,
   -19,-13,-8,2, 7,1,-18,-15, -16,-2,-6,22, -18,-12,-23,1, 0,-22,1,-8, 3,11,19,-6,
   11,4,-9,-9, 17,12,3,8, -19,-7,11,-13, -1,-14,-5,-5, 14,-6,-18,13, 9,19,20,-1,
   7,-10,12,15, -15,-4,18,14, 4,-1,7,-16, -18,-7,7,-8, 17,1,12,9, 6,18,-10,-15,
   -18,9,21,-11, 14,14,0,15, -4,-4,-1,1, 21,11,-18,-12, -4,16,14,12, 7,-15,-1,5,
   -12,-18,15,0, 23,-3,-8,-3, 20,-3,9,-2, -20,0,12,18, -9,-19,15,4, -12,-6,15,5,
   18,-1,20,-13, 1,10,7,4, -15,-11,2,-20, 0,-23,-16,12, 7,-6,-7,18, 17,8,-6,1,
   22,7,5,5, 13,-18,-20,-5, -10,6,9,12, 10,20,-6,17, 7,11,-10,-13, -3,8,-6,-4,
   -3,-3,0,-10, -9,-15,-14,12, 14,-7,0,13, -2,-15,12,17, -22,2,-7,-6, -6,-2,8,3,
   15,12,-5,18, -3,22,-10,-13, 23,-6,-15,-4, -5,7,12,13, -1,23,0,-4, 14,-10,4,-16,
   -8,-14,17,-5, 2,-2,-10,-19, -4,-5,15,0, 9,-9,7,-6, 9,-18,-1,5, 17,-11,-2,9,
   17,2,7,17, -13,14,16,-7, 13,-11,-3,18, 14,16,-1,13, 12,-2,5,18, -8,-7,15,-8,
   -7,-12,-6,-18, 19,-10,-6,22, -18,6,8,-14, -2,23,-6,6, -9,18,20,-11,
   -16,13,-16,-8, 4,-18,15,-7, 7,9,15,-18, -21,11,7,21, -6,9,6,6, 10,-2,2,12,
   -4,3,7,14, 20,6,0,-24, 7,-11,-6,13, 15,15,-20,-12, -11,17,-3,0, 12,-20,-3,-18,
   -11,15,14,-2, -5,-14,7,6, -10,20,1,-12, 9,9,3,-21, 0,16,-16,-6, -5,15,-5,0,
   16,15,-6,16, -1,-23,-5,-22, 19,3,12,-8, -7,-1,0,17, 12,-10,-21,0, -1,13,5,21,
   9,-13,11,-5, 6,7,-15,-14, -2,-2,5,9, -9,-13,-19,2, -9,14,16,14, 15,-2,-9,0,
   16,-5,-11,-18, -20,12,19,3, -6,-8,-3,0, -12,-13,-10,9, 11,3,11,19,
   -17,-12,7,11, -3,-6,11,12, -5,-5,-22,-5, 14,-14,-13,14, -11,10,6,-23,
   -6,-13,2,7, 12,7,13,11, -18,-14,-6,19, -3,2,9,-14, -1,16,20,13, 7,16,-9,8,
   11,-7,-4,-2, -4,-3,10,0, 7,17,-18,-1, 6,-4,2,-10, 2,-21,-18,12, 10,4,-14,7,
   -17,14,0,-24, 11,-11,-3,-10, -10,13,2,8, 9,7,-22,5, -14,9,-8,-2, 14,-17,11,-13,
   -5,-18,-1,0, 2,16,17,-16, 21,1,-17,10, 12,6,5,4, 10,14,10,-10, 23,0,10,14,
   10,8,-15,-8, 9,15,1,-9, -15,13,-3,-11, 6,-12,-4,-15, 19,-8,12,9, 12,13,1,14,
   1,-5,6,-20, -9,13,7,-3, -16,-17,14,-15, -18,5,9,13, 3,15,-21,0, 20,7,-13,-10,
   -21,-3,15,8, -10,-7,-18,-2, -4,-20,-10,-5, 6,-19,-12,-10, -7,-19,-20,5,
   4,13,19,-13, -5,-9,-5,15, -9,-10,-15,-3, -14,6,2,11, 10,11,6,-21, 5,-12,9,1,
   9,-7,-9,6, -6,-10,6,-8, -16,-5,17,-16, -1,7,11,18, 15,8,-11,-18, -19,-2,-14,-6,
   16,-1,6,-15, 22,2,-2,23, 4,-8,5,-23, -19,-8,-13,-2, 16,-6,-12,12, 3,17,-8,6,
   -2,3,8,3, 0,5,-4,13, 6,12,5,6, 7,12,21,1, -10,1,10,-2, 4,17,-5,7, -2,9,-2,21,
   0,-5,-6,-15, 5,4,16,-3, -10,-14,-15,1, 9,-10,6,-18, -2,4,11,3, 11,-1,18,15,
   3,-1,-19,-2, -16,11,-7,19, -13,-10,0,-8, 6,15,-9,21, 20,3,-7,17, -14,10,-20,-6,
   4,16,-2,-7, -8,3,21,3, 2,-16,6,3, -4,23,22,-2, -1,4,5,-4, -7,-7,-8,-9 };
          
    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        uchar* idesc = idescriptors.ptr(i);
        uchar* sdesc = sdescriptors.ptr(i);

        const cv::KeyPoint& pt = keypoints[i];
        double angle = pt.angle * DEGREE2RAD;

        cv::Mat R = (cv::Mat_<float>(2, 2) <<   cos(angle), -sin(angle), 
                                                sin(angle),  cos(angle));

        for( int j = 0; j < m_descriptor_size; j++ )
        {
            for (int k = 0; k < 8; k++)
            {
            	cv::Point2f p1, p2;
            	int index = j*8+k;

                p1.x = bit_pattern[index * 4]     * keypoints[i].response;
                p1.y = bit_pattern[index * 4 + 1] * keypoints[i].response;

                p2.x = bit_pattern[index * 4 + 2] * keypoints[i].response;
                p2.y = bit_pattern[index * 4 + 3] * keypoints[i].response;
            	
                cv::Mat P = (cv::Mat_<float>(2, 2) <<   p1.x, p2.x, 
                                                        p1.y, p2.y);
                P = R*P;

                p1.x = P.at<float>(0,0); p1.y = P.at<float>(1,0);
                p2.x = P.at<float>(0,1); p2.y = P.at<float>(1,1);

               int I1 = smoothedSum( sum, pt, p1 );
               int I2 = smoothedSum( sum, pt, p2 );

               idesc[j] += (I1 < I2) << (7-k);


               cv::Point3f pt1 = cloud.at<cv::Point3f>(p1.y, p1.x);
               cv::Point3f pt2 = cloud.at<cv::Point3f>(p2.y, p2.x);

	            if ( !std::isnan(pt1.x) &&  !std::isnan(pt1.y) && !std::isnan(pt1.z) &&
	                 !std::isnan(pt2.x) &&  !std::isnan(pt2.y) && !std::isnan(pt2.z) );
               {
                  cv::Point3f n1 = normals.at<cv::Point3f>(p1.y, p1.x);
                  cv::Point3f n2 = normals.at<cv::Point3f>(p2.y, p2.x);

                  bool dot_test = ( n1.dot(n2) <= m_degree_threshold );
                  bool convex_test = ( ( pt1 - pt2 ).dot( n1 - n2 ) < 0 );

                  sdesc[j] += (dot_test && convex_test) << (7-k);
                }
            }
        }
    }
}
