#define linux true

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <mutex>

std::mutex mutex_;
std::vector<cv::Point2f> prev_points_2D_;
std::vector<cv::Point2f> next_points_2D_;
std::vector<cv::Point3f> prev_points_3D_;
std::vector<cv::Point3f> next_points_3D_;
cv::Mat image_;

// ######################################################################
void openni_callback(
    boost::shared_ptr<openni_wrapper::Image> const & ni_image,
    boost::shared_ptr<openni_wrapper::DepthImage> const & ni_depth_image,
    float constant)
{
  static cv::Mat prev_image_gray;
  static std::vector<cv::Point2f> prev_points_2D;
  static std::vector<cv::Point3f> prev_points_3D;

  XnUInt64 const shadow_value    = ni_depth_image->getShadowValue();
  XnUInt64 const no_sample_value = ni_depth_image->getNoSampleValue();
  float const focal_length       = 1.0 / constant;

  // Copy the RGB image into a cv::Mat
  cv::Mat image(ni_image->getHeight(), ni_image->getWidth(), CV_8UC3);
  ni_image->fillRGB( ni_image->getWidth(), ni_image->getHeight(),
      reinterpret_cast<uint8_t*>(&image.begin<uint8_t>()[0]));

  float const cx = image.cols/2.0;
  float const cy = image.rows/2.0;

  // Convert the RGB image to grayscale
  cv::Mat image_gray;
  cv::cvtColor(image, image_gray, CV_BGR2GRAY);

  cv::Mat image_display;
  cv::cvtColor(image, image_display, CV_RGB2BGR);

  // Copy the depth image to a float Mat
  cv::Mat_<float> depth_image(ni_depth_image->getHeight(), ni_depth_image->getWidth());
  ni_depth_image->fillDepthImage(ni_depth_image->getWidth(), ni_depth_image->getHeight(), &depth_image.begin()[0]);

  std::vector<cv::Point3f> next_points_3D_clean;
  std::vector<cv::Point3f> prev_points_3D_clean;
  std::vector<cv::Point2f> next_points_2D_clean;
  std::vector<cv::Point2f> prev_points_2D_clean;
  if(prev_points_2D.size() > 0)
  {
    // Track the points detected from the last frame
    std::vector<cv::Point2f> next_points_2D;
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(prev_image_gray, image_gray, prev_points_2D, next_points_2D, status, error, cv::Size(5,5), 6);

    // Project all of the good track points to 3D
    for(size_t i=0; i<next_points_2D.size(); ++i)
    {
      if(!status[i]) continue;
      float next_depth = depth_image.at<float>(next_points_2D[i].y, next_points_2D[i].x);

      if(next_depth == shadow_value) continue;
      if(next_depth == no_sample_value) continue;

      cv::Point2f const & np = next_points_2D[i];
      cv::Point3f const next_point((np.x-cx)*next_depth/focal_length, (np.y-cy)*next_depth/focal_length, next_depth);

      next_points_3D_clean.push_back(next_point);
      //prev_points_3D_clean.push_back(prev_points_3D[i]);
      prev_points_2D_clean.push_back(prev_points_2D[i]);
      next_points_2D_clean.push_back(next_points_2D[i]);

      cv::circle(image_display, prev_points_2D[i], 5, cv::Scalar(0));
      cv::circle(image_display, next_points_2D[i], 5, cv::Scalar(0));
      cv::line(image_display, prev_points_2D[i], next_points_2D[i], cv::Scalar(128));
    }

    // Copy data for visualization
    {
      std::lock_guard<std::mutex> _(mutex_);
      prev_points_2D_ = prev_points_2D_clean;
      next_points_2D_ = next_points_2D_clean;
      prev_points_3D_ = prev_points_3D_clean;
      next_points_3D_ = next_points_3D_clean;
      image.copyTo(image_);
    }

    //// Find the affine transformation
    //cv::Mat affine_transform;
    //std::vector<uchar> outliers;
    //cv::estimateAffine3D(prev_points_3D_clean, next_points_3D_clean, affine_transform, inliers);
  }
  
  imshow("image", image_display);
  imshow("depth", depth_image/10);
  cv::waitKey(50);

  // Switch the new to the old
  image_gray.copyTo(prev_image_gray);
  prev_points_3D = next_points_3D_clean;

  // Detect trackable features in the next frame
  cv::goodFeaturesToTrack(prev_image_gray, prev_points_2D, 200, 0.01, 5);
}

// ######################################################################
int main(int argc, char** argv)
{
  pcl::Grabber * interface = new pcl::OpenNIGrabber();

  boost::function
    <
    void (
        boost::shared_ptr<openni_wrapper::Image> const &,
        boost::shared_ptr<openni_wrapper::DepthImage> const&,
        float)
    >
    openni_callback_func = boost::bind (openni_callback, _1, _2, _3);

  interface->registerCallback(openni_callback_func);
 
  interface->start(); 


  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Track Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  while(true)
  {
    viewer->spinOnce(100);
    usleep(50000);

    viewer->removePointCloud("track cloud");
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr visualization_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    visualization_cloud->clear();
    {
      std::lock_guard<std::mutex> _(mutex_);
      if(next_points_3D_.size() == 0) continue;

      for(int i=0; i<prev_points_2D_.size(); ++i)
      {
        pcl::PointXYZRGB next_point_3D;

        next_point_3D.x = next_points_3D_[i].x;
        next_point_3D.y = next_points_3D_[i].y;
        next_point_3D.z = next_points_3D_[i].z;

        cv::Vec3b c = image_.at<cv::Vec3b>(int(next_points_2D_[i].y), int(next_points_2D_[i].x));
        uint32_t const rgb = ((uint32_t)c[2] << 16 | (uint32_t)c[1] << 8 | (uint32_t)c[0]);
        next_point_3D.rgb = *reinterpret_cast<float const*>(&rgb);

        visualization_cloud->push_back(next_point_3D);
      }
    }
    visualization_cloud->height = 1;
    visualization_cloud->width = visualization_cloud->points.size();

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler(visualization_cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (visualization_cloud, rgb_handler, "track cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "track cloud");
  }

  return 0;
}
