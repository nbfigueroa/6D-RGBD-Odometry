/* 
   Copyright (C) 2013 Erickson R. Nascimento

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

#include <vector>
#include <algorithm>
//OPENCV
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

//PCL
#include <pcl/common/transforms.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> 


//Brand
#include "brand.h"
#include "evaluation_brand.h"

//Fovis
#include <stdint.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>


using namespace pcl;
using namespace std;
typedef pcl::ScopeTime ScopeTimeT;
typedef cv::Point_<float> Point2f;

const float fx = 525.0f;
const float fy = 525.0f;
const float cx = 319.5f;
const float cy = 239.5f;

double l1 (const Eigen::Vector4f &p_src, const Eigen::Vector4f &p_tgt) {
	return ((p_src.array () - p_tgt.array ()).abs ().sum ());
}

double  l2 (const Eigen::Vector4f &p_src, const Eigen::Vector4f &p_tgt){
      return ((p_src - p_tgt).norm ());
}


void crossCheckMatching( const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                         std::vector<cv::DMatch>& filteredMatches12 )
{
    //ScopeTimeT time ("crossCheckMatching");
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    filteredMatches12.clear();
    std::vector<std::vector<cv::DMatch> > matches12, matches21;
    matcher.knnMatch( descriptors1, descriptors2, matches12, 1 );
    matcher.knnMatch( descriptors2, descriptors1, matches21, 1 );
    for( size_t m = 0; m < matches12.size(); m++ )
    {
        bool findCrossCheck = false;
        for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
            cv::DMatch forward = matches12[m][fk];

            for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
                cv::DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if( findCrossCheck ) break;
        }
    }
}


void compute_normals(const cv::Mat& cloud, cv::Mat& normals)
{
   pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud( new pcl::PointCloud<pcl::PointXYZ> );
   pcl_cloud->clear();
   pcl_cloud->width     = cloud.cols;
   pcl_cloud->height    = cloud.rows;
   pcl_cloud->points.resize( pcl_cloud->width * pcl_cloud->height);
    
   for(int y = 0; y < cloud.rows; ++y)
   for(int x = 0; x < cloud.cols; ++x)
   {
      pcl_cloud->at(x,y).x = cloud.at<cv::Point3f>(y,x).x;
      pcl_cloud->at(x,y).y = cloud.at<cv::Point3f>(y,x).y;
      pcl_cloud->at(x,y).z = cloud.at<cv::Point3f>(y,x).z;
   }


   pcl::PointCloud<pcl::Normal>::Ptr pcl_normals (new pcl::PointCloud<pcl::Normal>);
   pcl_normals->clear();
   pcl_normals->width  = pcl_cloud->width;
   pcl_normals->height = pcl_cloud->height;
   pcl_normals->points.resize(pcl_cloud->width * pcl_cloud->height);

   pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
   ne.setInputCloud( pcl_cloud );

   ne.setNormalSmoothingSize( 5 );
   ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
   ne.compute( *pcl_normals );

   normals.create( cloud.size(), CV_32FC3 );

   for(int y = 0; y < pcl_normals->height; ++y)
   for(int x = 0; x < pcl_normals->width; ++x)
   {
      normals.at<cv::Point3f>(y,x).x = pcl_normals->at(x,y).normal_x;
      normals.at<cv::Point3f>(y,x).y = pcl_normals->at(x,y).normal_y; 
      normals.at<cv::Point3f>(y,x).z = pcl_normals->at(x,y).normal_z; 
   }
}

void create_cloud( const cv::Mat &depth, cv::Mat& cloud )
{
    //ScopeTimeT time ("Create Cloud");
    const float inv_fx = 1.f/fx;
    const float inv_fy = 1.f/fy;

    cloud.create( depth.size(), CV_32FC3 );

    for( int y = 0; y < cloud.rows; y++ )
    {
        cv::Point3f* cloud_ptr = (cv::Point3f*)cloud.ptr(y);
        const uint16_t* depth_prt = (uint16_t*)depth.ptr(y);

        for( int x = 0; x < cloud.cols; x++ )
        {
            float d = (float)depth_prt[x]/1000; // meters
            cloud_ptr[x].x = (x - cx) * d * inv_fx;
            cloud_ptr[x].y = (y - cy) * d * inv_fy;
            cloud_ptr[x].z = d;
        }
    }	
}


void extract_keypoints(cv::Mat& rgb1,cv::Mat& rgb2,std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2)
{
   //ScopeTimeT time ("Extract Keypoints");
   cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create( "STAR" );
   detector->detect( rgb1, keypoints1 );
   detector->detect( rgb2, keypoints2 );
}

void ransacOutlierRejection(cv::Mat& img1, std::vector<cv::KeyPoint>& keypoints1,cv::Mat& img2, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& filteredMatches, std::vector<cv::DMatch>& finalMatches, int ransacReprojThreshold, bool output){

    cv::Mat H12;
    vector<int> queryIdxs( filteredMatches.size() ), trainIdxs( filteredMatches.size() );
    for( size_t i = 0; i < filteredMatches.size(); i++ )
    {
       		 queryIdxs[i] = filteredMatches[i].queryIdx;
       		 trainIdxs[i] = filteredMatches[i].trainIdx;
    }

    cout << "< Computing homography (RANSAC)...";
    vector<Point2f> points1; cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
    vector<Point2f> points2; cv::KeyPoint::convert(keypoints2, points2, trainIdxs);
    H12 = cv::findHomography( cv::Mat(points1), cv::Mat(points2), CV_RANSAC, ransacReprojThreshold );
    cout << ">" << endl;
	
    cv::Mat drawImg;
    if( !H12.empty() ) 
    {
	// filter outliers on 2d
        std::vector<char> matchesMask( filteredMatches.size(), 0 );
        std::vector<Point2f> points1; cv::KeyPoint::convert(keypoints1, points1, queryIdxs);
        std::vector<Point2f> points2; cv::KeyPoint::convert(keypoints2, points2, trainIdxs);
        cv::Mat points1t; cv::perspectiveTransform(cv::Mat(points1), points1t, H12);

        double maxInlierDist = ransacReprojThreshold < 0 ? 3 : ransacReprojThreshold;
	for( size_t i1 = 0; i1 < points1.size(); i1++ )
        {
            if( norm(points2[i1] - points1t.at<Point2f>((int)i1,0)) <= maxInlierDist ) // inlier
		{
                matchesMask[i1] = 1;
		finalMatches.push_back(filteredMatches[i1]);

		}
	}	

	if (cv::countNonZero(matchesMask)>0){
        	if (output){
			// draw inliers
			cv::drawMatches( img1, keypoints1, img2, keypoints2, finalMatches, drawImg, CV_RGB(0, 255, 0), CV_RGB(0, 0, 255));
			cv::imshow( "Matches after Ransac Outlier Rejection", drawImg );
		}
	}
	else{
		if (output){
		finalMatches = filteredMatches;
	        drawMatches( img1, keypoints1, img2, keypoints2, finalMatches, drawImg );
        	cv::imshow( "Matches after Ransac Outlier Rejection", drawImg );
		cout << "No outliers rejected!!!" ;
		}
    	}
    }
	cv::waitKey();
}

void brand_matching(cv::Mat& rgb1, cv::Mat& cloud1, cv::Mat& rgb2, cv::Mat& cloud2, std::vector<cv::KeyPoint>& keypoints1, std::vector<cv::KeyPoint>& keypoints2, std::vector<cv::DMatch>& matches){

   //ScopeTimeT time ("Brand Matching");

   // detect keypoints using rgb images
   extract_keypoints(rgb1, rgb2, keypoints1, keypoints2);

   // create point clouds and compute normals
   cv::Mat normals1, normals2;
   compute_normals(cloud1, normals1);
   compute_normals(cloud2, normals2);

   // extract descriptors
   BrandDescriptorExtractor brand;
   cv::Mat desc1;
   brand.compute( rgb1, cloud1, normals1, keypoints1, desc1 );
   cv::Mat desc2;
   brand.compute( rgb2, cloud2, normals2, keypoints2, desc2 );

   // matching descriptors
   crossCheckMatching(desc2, desc1, matches);
}


void createPointcloudFromRegisteredDepthImage(cv::Mat& depthImage, cv::Mat& rgbImage, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& outputPointcloud)
{
	float rgbFocalInvertedX = 1/fx;	// 1/fx	
	float rgbFocalInvertedY = 1/fy;	// 1/fy

	outputPointcloud->clear();
        outputPointcloud->width     = depthImage.cols;
   	outputPointcloud->height    = depthImage.rows;
   	outputPointcloud->points.resize( outputPointcloud->width * outputPointcloud->height);
    
	pcl::PointXYZRGB newPoint;
	for (int i=0;i<depthImage.rows;i++)
		for (int j=0;j<depthImage.cols;j++){
			if (depthImage.at<uint16_t>(i,j) == depthImage.at<uint16_t>(i,j))                // if depthValue is not NaN
			{
				float d = (float) depthImage.at<uint16_t>(i,j)/(1000);
      				outputPointcloud->at(j,i).x = (float)(j - cx) * d * rgbFocalInvertedX;
			        outputPointcloud->at(j,i).y = (float)(i - cy) * d * rgbFocalInvertedY;
			        outputPointcloud->at(j,i).z = d;
				outputPointcloud->at(j,i).r = rgbImage.at<cv::Vec3b>(i,j)[2];
				outputPointcloud->at(j,i).g = rgbImage.at<cv::Vec3b>(i,j)[1];
				outputPointcloud->at(j,i).b = rgbImage.at<cv::Vec3b>(i,j)[0];		
			}
			else
			{
      				outputPointcloud->at(j,i).x = std::numeric_limits<float>::quiet_NaN();
			        outputPointcloud->at(j,i).y = std::numeric_limits<float>::quiet_NaN();
			        outputPointcloud->at(j,i).z = std::numeric_limits<float>::quiet_NaN();
				outputPointcloud->at(j,i).r = std::numeric_limits<unsigned char>::quiet_NaN();
				outputPointcloud->at(j,i).g = std::numeric_limits<unsigned char>::quiet_NaN();
				outputPointcloud->at(j,i).b = std::numeric_limits<unsigned char>::quiet_NaN();
			}
		}
}

void estimateRigidMotionSVD(std::vector<cv::DMatch>& final_matches, cv::Mat& rgb1, cv::Mat& depth1, std::vector<cv::KeyPoint>& keypoints1, cv::Mat& rgb2, cv::Mat& depth2, std::vector<cv::KeyPoint>& keypoints2, Eigen::Matrix4f& R, bool output){
 
    vector<int> queryIdxs( final_matches.size() ), trainIdxs( final_matches.size() );
    for( size_t i = 0; i < final_matches.size(); i++ ){
       		 queryIdxs[i] = final_matches[i].queryIdx;
       		 trainIdxs[i] = final_matches[i].trainIdx;
    }
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_rgb_cloud1 (new pcl::PointCloud<pcl::PointXYZRGB> ), pcl_rgb_cloud2 (new pcl::PointCloud<pcl::PointXYZRGB> );
    createPointcloudFromRegisteredDepthImage(depth1, rgb1, pcl_rgb_cloud1);
    createPointcloudFromRegisteredDepthImage(depth2, rgb2, pcl_rgb_cloud2);
    vector<Point2f> points1; cv::KeyPoint::convert(keypoints1, points1, trainIdxs);
    vector<Point2f> points2; cv::KeyPoint::convert(keypoints2, points2, queryIdxs);


	    vector <double> l1norms, l2norms;
	    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_src, pointcloud_tgt;
	    for( size_t i1 = 0; i1 < points1.size(); i1++ ){
			int id1_x(points1[i1].x), id1_y(points1[i1].y),id2_x(points2[i1].x), id2_y(points2[i1].y);
			pcl::PointXYZRGB p_src, p_tgt;
			// filter out idxs that don't have a valid depth value
			if((pcl_rgb_cloud1->at(id1_x,id1_y).z!=0)and(pcl_rgb_cloud2->at(id2_x,id2_y).z!=0)){
			p_src.x = pcl_rgb_cloud1->at(id1_x,id1_y).x;
			p_src.y = pcl_rgb_cloud1->at(id1_x,id1_y).y;
			p_src.z = pcl_rgb_cloud1->at(id1_x,id1_y).z;
			pointcloud_src.push_back(p_src);
			p_tgt.x = pcl_rgb_cloud2->at(id2_x,id2_y).x;
			p_tgt.y = pcl_rgb_cloud2->at(id2_x,id2_y).y;
			p_tgt.z = pcl_rgb_cloud2->at(id2_x,id2_y).z;
			pointcloud_tgt.push_back(p_tgt);
			Eigen::Vector4f s(p_src.x,p_src.y,p_src.z,1);
			Eigen::Vector4f t(p_tgt.x,p_tgt.y,p_tgt.z,1);
			//cout << "Match " << i1+1 << " l1: " << l1(s, t) << " l2: " << l2(s, t) << endl;
			l1norms.push_back(l1(s, t));l2norms.push_back(l2(s, t));

			}
		}
			
		double suml1, suml2;
		for(std::vector<double>::iterator j=l1norms.begin();j!=l1norms.end();++j)
    		   suml1 += *j;
		
		for(std::vector<double>::iterator j=l2norms.begin();j!=l2norms.end();++j)
    		   suml2 += *j;

		double meanl1 = suml1 / l1norms.size();
		double meanl2 = suml2 / l2norms.size();
            	pcl::PointCloud<pcl::PointXYZRGB> finalcloud_src, finalcloud_tgt;
	    	for(size_t i = 0; i < l1norms.size();++i){
			if (l1norms[i] < meanl1*1.75 and l2norms[i] < meanl2*1.75){
				finalcloud_src.push_back(pointcloud_src[i]);
				finalcloud_tgt.push_back(pointcloud_tgt[i]);
		    }
		}
		cout << "FIN@@L Corresponding Inliers: " << finalcloud_src.size() << endl;
	    if(output){    	  	
	    	 // --------------------------------------------
	   	 // -----Open 3D viewer and add point cloud-----
	    	 // --------------------------------------------
	    	 boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	    	 viewer->setBackgroundColor (1, 1, 1);
	    	 pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pcl_rgb_cloud1);
	    	 viewer->addPointCloud<pcl::PointXYZRGB> (pcl_rgb_cloud1, rgb, "sample cloud");
	    	 viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
            	 viewer->addPointCloud<pcl::PointXYZRGB> (pcl_rgb_cloud2, rgb, "sample2 cloud");
	    
		  for (size_t i=0;i<finalcloud_src.size();++i){	
	        		std::stringstream ss ("line"); ss << i;   
		        	std::stringstream sss ("spheresource"); sss << i;   
				std::stringstream ssss ("spheretarget");ssss << i;		
				viewer->addSphere<pcl::PointXYZRGB>(finalcloud_src[i],0.01,255,0,0,sss.str());
		        	viewer->addSphere<pcl::PointXYZRGB>(finalcloud_tgt[i],0.01,0,255,0,ssss.str());
		    	        viewer->addLine<pcl::PointXYZRGB> (finalcloud_src[i], finalcloud_tgt[i], 0, 255, 255, ss.str ());
			}
	    	 viewer->addCoordinateSystem (0.5);
	    	 viewer->resetCamera ();
	    	 viewer->spin();
	    }
	 
}



/*
void saveAllPoses(int frame_number, const std::string& logfile, vector<Eigen::Matrix4f> camera_poses) 
{   
  cout << "Writing " << frame_number << " poses to " << logfile << endl;
  
  ofstream path_file_stream(logfile.c_str());
  path_file_stream.setf(ios::fixed,ios::floatfield);
  
  for(int i = 0; i < frame_number; ++i)
  {
    Eigen::Affine3f pose = camera_poses[i];
    Eigen::Quaternionf q(pose.rotation());
    Eigen::Vector3f t = pose.translation();

    //double stamp = accociations_.empty() ? depth_stamps_and_filenames_[i].first : accociations_[i].time1;

    //path_file_stream << stamp << " ";
    path_file_stream << t[0] << " " << t[1] << " " << t[2] << " ";
    path_file_stream << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
  }
}


*/

int main(int argc, char** argv)
{

  if (argc != 2 )
   {
      std::cerr << "Usage: " << argv[0] << " dataset_directory.\n";
      return(1);
   }

   Evaluation::Ptr evaluation_ptr_;
   const std::string eval_folder = argv[1];
   evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder) );		
   int frames (evaluation_ptr_->getSizeAssociations());
   cout << frames << " frames." << endl;
	
   std::vector <std::vector<double> > camera_motion;
   camera_motion.resize(frames-1);
 //  for (size_t stamp_idx (830); stamp_idx < (frames -1)   ; ++stamp_idx){ 
  for (size_t stamp_idx (0); stamp_idx < (1)   ; ++stamp_idx){ 
           cout << "----- Frame " << stamp_idx+1 << " -----"<< endl;
           ScopeTimeT time ("RGB-D Visual Odometry");
       	   cv::Mat depth1, depth2, rgb1, rgb2, cloud1, cloud2;
           // grab frames from dataset folder
	   evaluation_ptr_->grab (stamp_idx, depth1, rgb1);	
	   evaluation_ptr_->grab (stamp_idx + 1, depth2, rgb2);
	   // create 3d point cloud from depth images
	   create_cloud( depth1, cloud1 );create_cloud( depth2, cloud2 );
           // convert 3d point cloud to pcl format
	   //Mat2PCD(cloud1, pcl_cloud1);Mat2PCD(cloud2, pcl_cloud2);
	   // define keypoints and matches
   	   std::vector<cv::KeyPoint> keypoints1, keypoints2;
	   std::vector<cv::DMatch> matches, newmatches;
	   // match brand features
   	   brand_matching(rgb1, cloud1, rgb2, cloud2, keypoints1, keypoints2, matches);
	   //brand_matching(rgb1, cloud1, pcl_cloud1, rgb2, cloud2, pcl_cloud2, keypoints1, keypoints2, matches);
	   cout << matches.size() <<" Inliers after cross-checking " << endl;
	   //-- Quick calculation of max and min distances between keypoints
	  double max_dist = 0; double min_dist = 0;
	  for( int i = 0; i < matches.size(); i++ )
	  { double dist = matches[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
	  }
	  printf("-- Max dist : %f \n", max_dist );
	  printf("-- Min dist : %f \n", min_dist );
	  //-- Use only "good" matches (i.e. whose distance is less than max_dist/2)
	  std::vector< cv::DMatch > good_matches, final_matches;
	  double reprojectionErrorThreshold (max_dist*3/4);
  	  for( int i = 0; i < matches.size(); i++ ) 
		if( matches[i].distance <  reprojectionErrorThreshold)
	       		good_matches.push_back( matches[i]); 
	  if (good_matches.size()>3)
	  	cout << good_matches.size() <<" \"Good matches\"" << endl;
          else{
		cout << "Too few good matches using previously filtered matches" << endl;
		good_matches = matches;
 	  }
	  //outlier rejection with RANSAC reprojection error and motion estimation	   
	  if (good_matches.size() > 3){
	  ransacOutlierRejection(rgb2, keypoints2, rgb1, keypoints1, good_matches, final_matches, reprojectionErrorThreshold, true);
	  cout << "Final number of inliers: " << final_matches.size()<< endl;
          // Estimate Rigid Transform by SVD on 3d points 
          Eigen::Matrix4f R;
	  estimateRigidMotionSVD(final_matches, rgb1, depth1, keypoints1, rgb2, depth2, keypoints2, R, true);
	  //vector<double> 6dmotion = rodrigues(R);
	  //camera_motion[stamp_idx] = 6dmotion;
    	  }
       	  else{
		cout << "Too few keypoints to compute homography, assuming motion from last two frames n-1("<< stamp_idx << ") and n(" << (stamp_idx-1) << ")"<<endl;
	   //camera_motion[stamp_idx]=camera_motion[stamp_idx-1];
           }
           
           
   }


   return 0;
}

