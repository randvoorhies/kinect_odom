#define linux true

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/video.hpp>
#include <mutex>

std::mutex mutex_;
std::vector<cv::Point2f> prev_points_2D_;
std::vector<cv::Point2f> curr_points_2D_;
std::vector<cv::Point3f> prev_points_3D_;
std::vector<cv::Point3f> curr_points_3D_;
cv::Mat curr_image_;

// ######################################################################
void openni_callback(
    boost::shared_ptr<openni_wrapper::Image> const & ni_image,
    boost::shared_ptr<openni_wrapper::DepthImage> const & ni_depth_image,
    float constant)
{
  static cv::Mat prev_image_gray;
  static cv::Mat prev_depth_image;
  static std::vector<cv::Point2f> prev_points_2D;

  XnUInt64 const shadow_value    = ni_depth_image->getShadowValue();
  XnUInt64 const no_sample_value = ni_depth_image->getNoSampleValue();
  float const focal_length       = 1.0 / constant;

  // Copy the RGB image into a cv::Mat
  cv::Mat curr_image(ni_image->getHeight(), ni_image->getWidth(), CV_8UC3);
  ni_image->fillRGB( ni_image->getWidth(), ni_image->getHeight(),
      reinterpret_cast<uint8_t*>(&curr_image.begin<uint8_t>()[0]));

  float const cx = curr_image.cols/2.0;
  float const cy = curr_image.rows/2.0;

  // Convert the RGB image to grayscale
  cv::Mat curr_image_gray;
  cv::cvtColor(curr_image, curr_image_gray, CV_BGR2GRAY);

  cv::Mat curr_image_display;
  cv::cvtColor(curr_image, curr_image_display, CV_RGB2BGR);

  // Copy the depth image to a float Mat
  cv::Mat_<float> curr_depth_image(ni_depth_image->getHeight(), ni_depth_image->getWidth());
  ni_depth_image->fillDepthImage(ni_depth_image->getWidth(), ni_depth_image->getHeight(), &curr_depth_image.begin()[0]);

  if(not prev_points_2D.empty())
  {
    // Track the points detected from the last frame
    std::vector<cv::Point2f> curr_points_2D;
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(prev_image_gray, curr_image_gray, prev_points_2D, curr_points_2D, status, error, cv::Size(5,5), 6);

    // Clear out all of the tracks with a bad status or no depth data, and project the good points to 3D
    std::vector<cv::Point2f> curr_points_2D_clean;
    std::vector<cv::Point2f> prev_points_2D_clean;
    std::vector<cv::Point3f> curr_points_3D;
    std::vector<cv::Point3f> prev_points_3D;
    for(size_t i=0; i<curr_points_2D.size(); ++i)
    {
      if(!status[i]) continue;

      float const curr_depth = curr_depth_image.at<float>(int(curr_points_2D[i].y), int(curr_points_2D[i].x));
      float const prev_depth = prev_depth_image.at<float>(int(prev_points_2D[i].y), int(prev_points_2D[i].x));

      if(curr_depth == shadow_value) continue;
      if(curr_depth == no_sample_value) continue;
      if(prev_depth == shadow_value) continue;
      if(prev_depth == no_sample_value) continue;
      if(std::isnan(curr_depth)) continue;
      if(std::isnan(prev_depth)) continue;

      prev_points_2D_clean.push_back(prev_points_2D[i]);
      curr_points_2D_clean.push_back(curr_points_2D[i]);

      cv::circle(curr_image_display, prev_points_2D[i], 5, cv::Scalar(0));
      cv::circle(curr_image_display, curr_points_2D[i], 5, cv::Scalar(0));
      cv::line(curr_image_display, prev_points_2D[i], curr_points_2D[i], cv::Scalar(128));

      cv::Point2f const & np = curr_points_2D[i];
      cv::Point3f const curr_point((np.x-cx)*curr_depth/focal_length, (np.y-cy)*curr_depth/focal_length, curr_depth);
      curr_points_3D.push_back(curr_point);
 
      cv::Point2f const & pp = prev_points_2D[i];
      cv::Point3f const prev_point((pp.x-cx)*prev_depth/focal_length, (pp.y-cy)*prev_depth/focal_length, prev_depth);
      prev_points_3D.push_back(prev_point);

      //printf("(%0.3f %0.3f)[%f] -> (%0.3f %0.3f)[%f] :: [%0.3f %0.3f %0.3f] -> [%0.3f %0.3f %0.3f]\n",
      //    pp.x, pp.y, prev_depth, np.x, np.y, curr_depth,
      //    prev_point.x, prev_point.y, prev_point.z,
      //    curr_point.x, curr_point.y, curr_point.z);
    }
    //printf("-------------------------------\n\n");

    if(not curr_points_2D_clean.empty())
    {
      // Find the affine transformation
      cv::Mat affine_transform;
      std::vector<uchar> outliers;
      cv::estimateAffine3D(prev_points_3D, curr_points_3D, affine_transform, outliers, .1);

      printf("T: [%+0.4f, %+0.4f, %+0.4f] Outliers: %zu/%zu\n",
        affine_transform.at<double>(0,3),
        affine_transform.at<double>(1,3),
        affine_transform.at<double>(2,3),
        outliers.size(),
        prev_points_3D.size());
    }

    // Copy data for visualization
    {
      std::lock_guard<std::mutex> _(mutex_);
      prev_points_2D_ = prev_points_2D_clean;
      curr_points_2D_ = curr_points_2D_clean; 
      prev_points_3D_ = prev_points_3D;
      curr_points_3D_ = curr_points_3D;
      curr_image.copyTo(curr_image_);
    }
  }
  
  imshow("image", curr_image_display);
  imshow("depth", curr_depth_image/10);
  cv::waitKey(50);

  // Switch the new to the old
  curr_image_gray.copyTo(prev_image_gray);
  curr_depth_image.copyTo(prev_depth_image);

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
      if(curr_points_3D_.size() == 0) continue;

      for(int i=0; i<prev_points_2D_.size(); ++i)
      {
        pcl::PointXYZRGB curr_point_3D;

        curr_point_3D.x = curr_points_3D_[i].x;
        curr_point_3D.y = curr_points_3D_[i].y;
        curr_point_3D.z = curr_points_3D_[i].z;

        cv::Vec3b c = curr_image_.at<cv::Vec3b>(int(curr_points_2D_[i].y), int(curr_points_2D_[i].x));
        uint32_t const rgb = ((uint32_t)c[2] << 16 | (uint32_t)c[1] << 8 | (uint32_t)c[0]);
        curr_point_3D.rgb = *reinterpret_cast<float const*>(&rgb);

        visualization_cloud->push_back(curr_point_3D);
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
