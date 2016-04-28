#include <vector>
#include <algorithm>

//OPENCV
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>

//PCL
#include <pcl/common/transforms.h>
#include <pcl/common/distances.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/common/time.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/registration/registration.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/common/eigen.h>
#include <boost/thread/thread.hpp>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/filters/voxel_grid.h>

//Brand
#include "brand/brand.h"
#include "evaluation_brand.h"

//Fovis
#include <stdint.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>


typedef pcl::PointXYZ PointType;

using namespace pcl;
using namespace std;
typedef pcl::ScopeTime ScopeTimeT;
typedef cv::Point_<float> Point2f;

const float fx = 525.0f;
const float fy = 525.0f;
const float cx = 319.5f;
const float cy = 239.5f;

// --------------------
// -----Parameters-----
// --------------------
float angular_resolution = 0.5f;
//float support_size = 0.1f;
float support_size = 0.2f;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
bool setUnseenToMaxRange = false;
int max_no_of_threads = 8;
float min_interest_value = 0.1;
typedef std::pair <int, int> Intpair;
vector<Intpair> absolute_indices;

struct FrameComputation{
   cv::Mat cloud, normals, desc;
   std::vector<cv::KeyPoint> keypoints;
   pcl::PointCloud<PointXYZ>::Ptr pcl_cloud;
   pcl::PointCloud<pcl::Normal>::Ptr pcl_normals ;
   
   FrameComputation(){} 

   FrameComputation(cv::Mat& cloud_, cv::Mat& normals_, std::vector<cv::KeyPoint>& keypoints_, cv::Mat& desc_, pcl::PointCloud<PointXYZ>::Ptr& pcl_cloud_, pcl::PointCloud<pcl::Normal>::Ptr& pcl_normals_){
	cloud = cloud_;
	normals = normals_;
	keypoints = keypoints_;
	desc = desc_;
	pcl_cloud = pcl_cloud_;
	pcl_normals = pcl_normals_;
   } 

   void assign(cv::Mat& cloud_, cv::Mat& normals_, std::vector<cv::KeyPoint>& keypoints_, cv::Mat& desc_, pcl::PointCloud<PointXYZ>::Ptr& pcl_cloud_, pcl::PointCloud<pcl::Normal>::Ptr& pcl_normals_){
  	cloud = cloud_;
	normals = normals_;
	keypoints = keypoints_;
	desc = desc_;
	pcl_cloud = pcl_cloud_;
	pcl_normals = pcl_normals_;
   }
};

vector<FrameComputation> framecomputations;
pcl::PointCloud<PointXYZRGB> model;

pcl::visualization::PCLVisualizer *viewer2 = new pcl::visualization::PCLVisualizer ("Reconstruction Viewer");
int globindx (0);


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


void compute_normals( cv::Mat& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud, cv::Mat& normals, pcl::PointCloud<pcl::Normal>::Ptr& pcl_normals)
{
   pcl_normals->clear();
   pcl_normals->width  = pcl_cloud->width;
   pcl_normals->height = pcl_cloud->height;
   pcl_normals->points.resize(pcl_cloud->width * pcl_cloud->height);

   pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
   ne.setInputCloud( pcl_cloud );

   ne.setNormalSmoothingSize( 10 );
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

//---- Extract STAR Keypoints ----//
//void extract_keypoints(cv::Mat& rgb1,cv::Mat& rgb2,std::vector<cv::KeyPoint>& keypoints1,std::vector<cv::KeyPoint>& keypoints2)
//{
//   //ScopeTimeT time ("Extract Keypoints");
//   cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create( "STAR" );
//   detector->detect( rgb1, keypoints1 );
//   detector->detect( rgb2, keypoints2 );
//}


void convert_cv2pcl_cloud(cv::Mat& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& pcl_cloud){

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
}

void 
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2]);
}


void extractNARFkeypoints(cv::Mat& cloud,   std::vector<cv::KeyPoint>& keypoints_narf){ 
 
      pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud (new pcl::PointCloud<pcl::PointXYZ>);
      convert_cv2pcl_cloud(cloud, pcl_cloud);
      pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;
      Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
  // -----------------------------------------------
  // -----Create RangeImage from the DepthImage-----
  // -----------------------------------------------

      std::vector<float> source_depth_data_;
      int width = cloud.cols, height = cloud.rows;
      source_depth_data_.resize(width * height);
      float *depth_buffer = (float *) &source_depth_data_[0];  

      //std::cout << "Giving colors3\n";
      for (int i=0; i<width*height; i++) {
        depth_buffer[i]    = pcl_cloud->points[i].z;
      }

      float noise_level = 0.0;
      float min_range = 0.0f;
      int border_size = 0;
      boost::shared_ptr<pcl::RangeImagePlanar> range_image_ptr (new pcl::RangeImagePlanar);
      pcl::RangeImagePlanar& range_image_planar = *range_image_ptr;   
      float center_x = width/2, center_y = height/2;
      float original_angular_resolution = asinf (0.5f*float (width)/float (fx)) / (0.5f*float (width));
      float desired_angular_resolution = angular_resolution;
      range_image_planar.setDepthImage (&source_depth_data_[0], width, height, center_x, center_y, fx, fy);
      range_image_planar.setUnseenToMaxRange();

  // --------------------------------
  // -----Extract NARF keypoints-----
  // --------------------------------
      pcl::RangeImageBorderExtractor range_image_border_extractor;
      pcl::NarfKeypoint narf_keypoint_detector (&range_image_border_extractor);
      narf_keypoint_detector.setRangeImage (&range_image_planar);
      narf_keypoint_detector.getParameters ().support_size = support_size;
      narf_keypoint_detector.getParameters ().max_no_of_threads = max_no_of_threads;
      narf_keypoint_detector.getParameters ().min_interest_value = min_interest_value;
//      narf_keypoint_detector.getParameters ().add_points_on_straight_edges = true;
      narf_keypoint_detector.getParameters ().calculate_sparse_interest_image = true;
      narf_keypoint_detector.getParameters ().use_recursive_scale_reduction = true;
      
      pcl::PointCloud<int> keypoint_indices;
      double keypoint_extraction_start_time = pcl::getTime();
      narf_keypoint_detector.compute (keypoint_indices);
      double keypoint_extraction_time = pcl::getTime()-keypoint_extraction_start_time;
      std::cout << "Found "<<keypoint_indices.points.size ()<<" key points. "
              << "This took "<<1000.0*keypoint_extraction_time<<"ms.\n";

     
      // find corresponding index to keypoint 3D coords
      std::vector<cv::Point2f> keypoints_2d;
      for (size_t i=0; i<keypoint_indices.points.size (); ++i){
	     std::pair <int,int> p = absolute_indices.at(keypoint_indices.points[i]);
	     cv::Point2f point2d (p.second, p.first);
	     keypoints_2d.push_back(point2d);	 
      }
      cv::KeyPoint::convert(keypoints_2d, keypoints_narf);
     

}


struct Kgreater
{
    bool operator()( const cv::KeyPoint& k1, const cv::KeyPoint& k2 ) const {
    	return k1.pt.x < k2.pt.x and k1.pt.y < k2.pt.y;
    }
};

struct Ksame{
	bool operator()(const cv::KeyPoint& k1, const cv::KeyPoint& k2) const {
    	return k1.pt.x == k2.pt.x and k1.pt.y == k2.pt.y;
    }
};

void combineKeypoints( std::vector<cv::KeyPoint>& keypoints_star, std::vector<cv::KeyPoint>& keypoints_narf,  std::vector<cv::KeyPoint>& keypoints){
	keypoints = keypoints_star;
	keypoints.insert(keypoints.end(), keypoints_narf.begin(), keypoints_narf.end());
	std::sort( keypoints.begin(), keypoints.end(), Kgreater() );
        keypoints.erase( unique( keypoints.begin(), keypoints.end(), Ksame() ), keypoints.end() );
}

void extract_keypoints(cv::Mat& rgb, cv::Mat& cloud, std::vector<cv::KeyPoint>& keypoints)
{
   //Extract A* Keypoints (RGB)
   std::vector<cv::KeyPoint> keypoints_star; 	
   cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create( "STAR" );
   detector->detect( rgb, keypoints_star);

   //Extract NARF Keypoints (Depth)
   std::vector<cv::KeyPoint> keypoints_narf;
   pcl::PointCloud<int> keypoint_indices;
   extractNARFkeypoints(cloud, keypoints_narf);

   //Combine A* + NARF Keypoints
   combineKeypoints(keypoints_star, keypoints_narf, keypoints);
 
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
	cv::waitKey(1);
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


void convert2qt(Eigen::Matrix4f pose, Eigen::Vector3f& t, Eigen::Quaternionf& q){
 		
   //cout << pose << endl;
   Eigen::Matrix3f Rotation = pose.block(0,0,3,3); 
   //Rotation = Rotation.householderQr().householderQ();
   t = pose.block(0,3,3,1);
   Eigen::Quaternionf q2(Rotation);	
   q = q2;
   //cout << t[0] << " " << t[1] << " " << t[2] << " ";
   //cout << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
}



Eigen::Matrix4f rigidMotionEstimation(const pcl::PointCloud<PointXYZRGBNormal> &cloud_src,const pcl::PointCloud<PointXYZRGBNormal> &cloud_tgt, const pcl::Correspondences &correspondences){

  // Convert Points to Eigen Format!!
  ConstCloudIterator<PointXYZRGBNormal> source_it (cloud_src, correspondences, true);
  ConstCloudIterator<PointXYZRGBNormal> target_it (cloud_tgt, correspondences,true);
  const int npts = static_cast <int> (source_it.size ());

  Eigen::Matrix<float, 3, Eigen::Dynamic> cloud_src_eig (3, npts);
  Eigen::Matrix<float, 3, Eigen::Dynamic> cloud_tgt_eig (3, npts);

  for (int i = 0; i < npts; ++i)
  {
    cloud_src_eig (0, i) = source_it->x;
    cloud_src_eig (1, i) = source_it->y;
    cloud_src_eig (2, i) = source_it->z;
    ++source_it;

    cloud_tgt_eig (0, i) = target_it->x;
    cloud_tgt_eig (1, i) = target_it->y;
    cloud_tgt_eig (2, i) = target_it->z;
    ++target_it;
  }

  // Call Umeyama directly from Eigen (PCL patched version until Eigen is released)
  
  Eigen::Matrix4f transformation_matrix = pcl::umeyama (cloud_src_eig, cloud_tgt_eig, true);

  return transformation_matrix;
}


void estimateRigidMotion(std::vector<cv::DMatch>& final_matches, cv::Mat& rgb1, cv::Mat& depth1, pcl::PointCloud<pcl::Normal>::Ptr& pcl_normals1, std::vector<cv::KeyPoint>& keypoints1, cv::Mat& rgb2, cv::Mat& depth2, pcl::PointCloud<pcl::Normal>::Ptr& pcl_normals2, std::vector<cv::KeyPoint>& keypoints2, Eigen::Matrix4f& T_prev, Eigen::Matrix4f& T_current, bool output){
 
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
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pcl_rgb_cloud1);
    pcl::transformPointCloud(*pcl_rgb_cloud1,*pcl_rgb_cloud1,T_prev);
    pcl::transformPointCloud(*pcl_rgb_cloud2,*pcl_rgb_cloud2,T_prev);
    

	if(globindx==0 and output){
		Eigen::Vector3f t; Eigen::Quaternionf q;
		std::stringstream sss_cl ("cloud"); sss_cl << globindx;
    		viewer2->addPointCloud<pcl::PointXYZRGB> (pcl_rgb_cloud1, rgb, sss_cl.str());
		convert2qt(T_prev,t,q);
		Eigen::Matrix3f R( q.toRotationMatrix());
		Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);
		viewer2->addCoordinateSystem(0.1,pose);
	}
	    vector <double> l1norms, l2norms;
	    pcl::PointCloud<pcl::PointXYZRGBNormal> pointcloud_src, pointcloud_tgt;
	    for( size_t i1 = 0; i1 < points1.size(); i1++ ){
			int id1_x(points1[i1].x), id1_y(points1[i1].y),id2_x(points2[i1].x), id2_y(points2[i1].y);
			pcl::PointXYZRGBNormal p_src, p_tgt;
			// filter out idxs that don't have a valid depth value
			if((pcl_rgb_cloud1->at(id1_x,id1_y).z!=0)and(pcl_rgb_cloud2->at(id2_x,id2_y).z!=0)){
			p_src.x = pcl_rgb_cloud1->at(id1_x,id1_y).x;
			p_src.y = pcl_rgb_cloud1->at(id1_x,id1_y).y;
			p_src.z = pcl_rgb_cloud1->at(id1_x,id1_y).z;
			p_src.normal_x = pcl_normals1->at(id1_x,id1_y).normal_x;
			p_src.normal_y = pcl_normals1->at(id1_x,id1_y).normal_y;
			p_src.normal_z = pcl_normals1->at(id1_x,id1_y).normal_z;
			p_src.curvature = pcl_normals1->at(id1_x,id1_y).curvature;
			pointcloud_src.push_back(p_src);
			p_tgt.x = pcl_rgb_cloud2->at(id2_x,id2_y).x;
			p_tgt.y = pcl_rgb_cloud2->at(id2_x,id2_y).y;
			p_tgt.z = pcl_rgb_cloud2->at(id2_x,id2_y).z;
			pointcloud_tgt.push_back(p_tgt);
			Eigen::Vector4f s(p_src.x,p_src.y,p_src.z,1);
			Eigen::Vector4f t(p_tgt.x,p_tgt.y,p_tgt.z,1);
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
		pcl::Correspondences final_correspondences; 
           	pcl::PointCloud<pcl::PointXYZRGBNormal> finalcloud_src, finalcloud_tgt;
		// Fill in corresponding pairs ...	
		cout << "FIN@@L Corresponding Inliers: " << endl;
	    	double maxl1 (0.0);
		for(size_t i = 0; i < l1norms.size();++i){
			if (l1norms[i] < meanl1*1.5 and l2norms[i] < meanl2*1.5){
				finalcloud_src.push_back(pointcloud_src[i]);
				finalcloud_tgt.push_back(pointcloud_tgt[i]);
				pcl::Correspondence corr;
				corr.index_query = i; corr.index_match=i; corr.distance = l2norms[i]; 
				//cout  << corr << endl;
				if(l1norms[i] > maxl1)
					maxl1 = l1norms[i];
				final_correspondences.push_back(corr);
		    }
		}
		cout << "Total:"<< final_correspondences.size() << endl;
		cout << "Maximum Correspondence Distance: " << maxl1 << endl; 

	   // Estimate rigid motion (6d) ... iterative or exact?
	   // Obtain the best transformation between the two sets of keypoints given the remaining correspondences
		
  	   float final_error = 0.50, threshold_(0.003), error_threshold (0.10), hypotheses_iterations(200);
	   int c (0), k(5);
	   Eigen::Matrix4f T;
	   std::vector<int> indices, combo;
	   vector<Eigen::Matrix4f> transforms;
	   vector<float> errors;
	   for(int i=0;i<final_correspondences.size();++i)
			indices.push_back(i);

           bool min_found(false);
	   while (c < hypotheses_iterations){
		c++;		
		// Find a random combination (k corresponding pairs)
		combo.resize(k);
		std::random_shuffle(indices.begin(),indices.end());
		copy(indices.begin(),indices.begin()+k,combo.begin());
		pcl::Correspondences iter_correspondences; 
		// Fill in corresponding pairs ...	
		for(size_t i = 0; i < combo.size();++i){
				pcl::Correspondence corr;
				corr.index_query = combo[i]; corr.index_match=combo[i]; corr.distance = final_correspondences[combo[i]].distance;
				iter_correspondences.push_back(corr);
		    }

		T = rigidMotionEstimation(finalcloud_src, finalcloud_tgt, iter_correspondences);		
		T = T.inverse();
		pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr transformed (new pcl::PointCloud<pcl::PointXYZRGBNormal> );
		pcl::transformPointCloud(finalcloud_tgt,*transformed,T);
		std::vector<int> nn_index (1);
  		std::vector<float> nn_distance (1);
		float error (0.0);
		pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree_ (new pcl::search::KdTree<pcl::PointXYZRGBNormal> ());
		tree_->setInputCloud (finalcloud_src.makeShared());
		for (int i = 0; i < static_cast<int> (transformed->points.size ()); ++i){
		    // Find the distance between cloud.points[i] and its nearest neighbor in the target point cloud
		    tree_->nearestKSearch (*transformed, i, 1, nn_index, nn_distance);
		    // Compute the error
		    float e (nn_distance[0]), e_(0);
		     if (e <= threshold_)
			e_= (e / threshold_);
		     else
			e_=1.0;
		    error += e_;
  		}
		final_error = error/transformed->points.size ();
		if(final_error>0){
			errors.push_back(final_error);
			transforms.push_back(T);
		}


		if(c>(hypotheses_iterations-5)){
			hypotheses_iterations = hypotheses_iterations + 100;
			k = k+1;
	        }
   		if(c>1000)
		  break;
	        combo.clear();
	   }


		float min_error = *std::min_element(errors.begin(), errors.end());
		std::vector<float>::iterator  min_error_it = std::min_element(errors.begin(), errors.end());
		int min_error_id = static_cast<int> (min_error_it - errors.begin());
		std::cout << "Minimum error of " << min_error << " at iteration " << min_error_id << endl;
		Eigen::Matrix4f min_T = transforms[min_error_id];
		Eigen::Vector3f t_r; Eigen::Quaternionf q_r; Eigen::Vector3f rpy = Eigen::Isometry3f(min_T).rotation().eulerAngles(0,1,2);
		convert2qt(min_T,t_r,q_r);
		std::cout << "Relative translation:" << endl;
		double relative_t = sqrt(t_r[0]*t_r[0] +  t_r[1]*t_r[1] + t_r[2]*t_r[2]);
 		std:: cout << "abs t: " << relative_t << " r: " << (rpy[0]*180)/3.14159 << " p: " << (rpy[1]*180)/3.14159 << " y: " << (rpy[2]*180)/3.14159 << endl;
		double rotation_threshold (10.0), translation_threshold (0.15), absolute_r;
		absolute_r = abs((rpy[0]*180)/3.14159) + abs((rpy[1]*180)/3.14159) + abs((rpy[2]*180)/3.14159);
		if (relative_t > translation_threshold or absolute_r > rotation_threshold)
			cv::waitKey();  


		//ICP-NL Refinement
// 		pcl::transformPointCloud(finalcloud_tgt,finalcloud_tgt,min_T);
//		pcl::IterativeClosestPointNonLinear<pcl::PointXYZRGBNormal,pcl::PointXYZRGBNormal> icp_nl;		
//		icp_nl.setTransformationEpsilon (1e-8); icp_nl.setMaxCorrespondenceDistance (0.003); 
//		icp_nl.setInputSource (finalcloud_src.makeShared()); icp_nl.setInputTarget (finalcloud_tgt.makeShared());
// 		pcl::PointCloud<pcl::PointXYZRGBNormal> Final;
// 	 	icp_nl.align(Final);
//  		T = icp_nl.getFinalTransformation();

//		std::cout << "After ICP-NL refinement: " << endl;
//		Eigen::Matrix4f T_ref = min_T*T.inverse();
//		std::cout << T_ref << endl;


		// After final transformation
//		cout << "Final Error: " << final_error << endl;
//		cout << "Estimated Transformation after " << c << " iterations:" << endl; 
		Eigen::Matrix4f T_ref = min_T; //no refinement
		T_current = T_ref*T_prev;
		Eigen::Vector3f t; Eigen::Quaternionf q;
		convert2qt(T_current,t,q);
//   		cout << t[0] << " " << t[1] << " " << t[2] << " ";
//   		cout << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl

     		//--Visualization


		if (output){
		pcl::transformPointCloud(*pcl_rgb_cloud2,*pcl_rgb_cloud2,T_ref);
		std::stringstream sss_cl ("ccloud"); sss_cl << globindx;
    		viewer2->addPointCloud<pcl::PointXYZRGB> (pcl_rgb_cloud2, rgb, sss_cl.str());
		Eigen::Matrix3f R( q.toRotationMatrix());
		Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);
		viewer2->addCoordinateSystem(0.1,pose);
	    	viewer2->resetCamera ();
	    	viewer2->spinOnce();

		}
	
	    globindx++;
	 
}



void saveAllPoses(int frame_number, vector<Eigen::Matrix4f> camera_poses) 
{   
  string logfile("6d_odometry_poses.txt");
  cout << "Writing " << frame_number << " poses to " << logfile << endl;
  ofstream path_file_stream(logfile.c_str());
  path_file_stream.setf(ios::fixed,ios::floatfield);
  
  for(int i = 0; i < frame_number; ++i)
  {
    Eigen::Quaternionf q;
    Eigen::Vector3f t;
    convert2qt(camera_poses[i],t,q);

    //double stamp = accociations_.empty() ? depth_stamps_and_filenames_[i].first : accociations_[i].time1;

    path_file_stream << i << " ";
    path_file_stream << t[0] << " " << t[1] << " " << t[2] << " ";
    path_file_stream << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
  }
}


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
//   frames = frames - 10;
//   cout << " Trajectory of " << frames << " frames.\n";
   frames = 1000;
//   viewer2->setBackgroundColor (1, 1, 1);
   std::vector <Eigen::Matrix4f> camera_motion;
   camera_motion.resize(frames);
   
   //-- Initialize absolute indices of 640x480 images
   for(int y = 0; y < 480; ++y)
   for(int x = 0; x < 640; ++x){
      std::pair <int, int> pixel_pair = std::make_pair( y, x );
      absolute_indices.push_back(pixel_pair);
   }

   // ------------------------------------------------------//
   // -------------- MAIN PROCESSING LOOP ------------------//
   // ------------------------------------------------------//

//   for (size_t stamp_idx (85); stamp_idx < (86)   ; ++stamp_idx){ 
   for (size_t stamp_idx (0); stamp_idx < (frames-1)   ; ++stamp_idx){ 

           cout << "----- Frame " << stamp_idx+1 << " -----"<< endl;
           ScopeTimeT time ("RGB-D Visual Odometry");
       	   cv::Mat depth1, depth2, rgb1, rgb2, cloud1, cloud2;
           //-- Grab frames from dataset folder
	   evaluation_ptr_->grab (stamp_idx, depth1, rgb1);	
	   evaluation_ptr_->grab (stamp_idx + 1, depth2, rgb2);

           // ------------------------------------------------------ //
	   // ---- Initialization/Creation of clouds,pcl,normals --- //
   	   // ------------------------------------------------------ //
	   pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud1 (new pcl::PointCloud<pcl::PointXYZ>), pcl_cloud2 (new pcl::PointCloud<pcl::PointXYZ>);
	   pcl::PointCloud<pcl::Normal>::Ptr normals1pcl (new pcl::PointCloud<pcl::Normal>), normals2pcl (new pcl::PointCloud<pcl::Normal>);
           cv::Mat normals1, normals2;

   	   // ------------------------------------------------------//
           // ----- Keypoint Detection and Feature Computation -----//
	   // ------------------------------------------------------//
    	   //-- Define keypoints and matches
   	   std::vector<cv::KeyPoint> keypoints1, keypoints2;
   	   cv::Mat desc1, desc2;
	   BrandDescriptorExtractor brand;
           FrameComputation frame1, frame2;

	   if(stamp_idx == 0){
           	create_cloud( depth1, cloud1 );
	   	convert_cv2pcl_cloud(cloud1, pcl_cloud1);
	        compute_normals(cloud1, pcl_cloud1, normals1, normals1pcl);
   	   	extract_keypoints(rgb1, cloud1, keypoints1);
   	   	brand.compute( rgb1, cloud1, normals1, keypoints1, desc1 );
	        frame1.assign(cloud1, normals1, keypoints1, desc1, pcl_cloud1, normals1pcl);
		framecomputations.push_back(frame1);	
	   }
	   else
		frame1.assign (framecomputations[stamp_idx].cloud, framecomputations[stamp_idx].normals, framecomputations[stamp_idx].keypoints, framecomputations[stamp_idx].desc, framecomputations[stamp_idx].pcl_cloud, framecomputations[stamp_idx].pcl_normals);

	   //-- Create 3d point cloud from depth images
	   create_cloud( depth2, cloud2 );
	   //-- Convert 3d point cloud to pcl format
	   convert_cv2pcl_cloud(cloud2, pcl_cloud2);
	   //-- Compute normals in opencv and pcl format		
           compute_normals(cloud2, pcl_cloud2, normals2, normals2pcl);
   	   //-- Extract keypoints using rgb (A*) and depth (NARF) images
	   extract_keypoints(rgb2, cloud2, keypoints2);
	   //-- Compute BRAND features
   	   brand.compute( rgb2, cloud2, normals2, keypoints2, desc2 );

           frame2.assign(cloud2, normals2, keypoints2, desc2, pcl_cloud2, normals2pcl);
	   framecomputations.push_back(frame2);

	   // ------------------------------------------------------//
           // -----  Feature Matching and Outlier Rejection --------//
	   // ------------------------------------------------------//
	   //-- Match features using cross-checking with Hamming norm
	   std::vector<cv::DMatch> matches; 
	   crossCheckMatching(frame2.desc, frame1.desc, matches);
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
	   double reprojectionErrorThreshold (max_dist*1/2);
  	   for( int i = 0; i < matches.size(); i++ ) 
		if( matches[i].distance <  reprojectionErrorThreshold)
	       		good_matches.push_back( matches[i]); 
	   if (good_matches.size()>3)
	  	cout << good_matches.size() <<" \"Good matches\"" << endl;
           else{
		cout << "Too few good matches using previously filtered matches" << endl;
		good_matches = matches;
 	   }
	   //-- Outlier rejection with RANSAC reprojection error and motion estimation	   
	   if (good_matches.size() > 30){
	   ransacOutlierRejection(rgb2, frame2.keypoints, rgb1, frame1.keypoints, good_matches, final_matches, reprojectionErrorThreshold, true);
	   cout << "Final number of inliers: " << final_matches.size()<< endl;


	  // ------------------------------------------------------//
          //----- Estimate Rigid Transform by SVD on 3d points ----//
          // ------------------------------------------------------//
	  Eigen::Matrix4f T, T_prev;
		  if (stamp_idx==0){
			// tx ty tz qx qy qz qw
			//-0.0730 -0.4169 1.5916 0.8772 -0.1170 0.0666 -0.4608
			//1305031910.7695 -0.8683 0.6026 1.5627 0.8219 -0.3912 0.1615 -0.3811
			Eigen::Vector4f t (-0.868270, 0.603138, 1.562795, 1);
			Eigen::Quaternionf q(-0.381461, 0.822052,-0.390615, 0.16170);

			Eigen::Matrix3f R( q.toRotationMatrix());					  
			//Eigen::Vector4f t (0, 0, 0, 1);
			//Eigen::Matrix3f R (Eigen::Matrix3f::Identity());
			  
			  T_prev.block(0,0,3,3) = R;			
			  T_prev.block(0,3,4,1) = t;
		          camera_motion[stamp_idx] = T_prev;
			  }
		  else
			T_prev = camera_motion[stamp_idx];

		   estimateRigidMotion(final_matches, rgb1, depth1, frame1.pcl_normals, frame1.keypoints, rgb2, depth2, frame2.pcl_normals, frame2.keypoints, T_prev, T, false);
	           camera_motion[stamp_idx+1] = T;  
    	  }
       	  else{
		cout << "Too few keypoints to compute camera pose, assuming motion from last two frames n-1("<< stamp_idx << ") and n(" << (stamp_idx-1) << ")"<<endl;
	   Eigen::Matrix4f T_0 (camera_motion[stamp_idx-1]);
	   Eigen:: Matrix4f T_1 (camera_motion[stamp_idx]);
	   Eigen::Matrix4f T_0_1 (T_1.inverse()*T_0);
	   Eigen::Matrix4f T_2 (T_1*T_0_1);
	   camera_motion[stamp_idx+1] = T_2;
	   cv::waitKey();
           }
   }
   cout << "Before save all poses" << endl;
   saveAllPoses(frames,camera_motion);
   viewer2->spin();
   //saveAllPoses(11,camera_motion);

   return 0;
}

