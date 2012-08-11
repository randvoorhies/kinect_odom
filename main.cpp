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


void openni_callback(
    boost::shared_ptr<openni_wrapper::Image> const & ni_image,
    boost::shared_ptr<openni_wrapper::DepthImage> const & ni_depth_image,
    float constant)
{
  static cv::Mat prev_image_gray;
  static std::vector<cv::Point2f> prev_points;
  static std::vector<cv::Point3f> prev_points_3D;

  XnUInt64 const shadow_value    = ni_depth_image->getShadowValue();
  XnUInt64 const no_sample_value = ni_depth_image->getNoSampleValue();
  float const focal_length       = 1.0 / constant;

  // Copy the RGB image into a cv::Mat
  cv::Mat image(ni_image->getHeight(), ni_image->getWidth(), CV_8UC3);
  ni_image->fillRGB( ni_image->getWidth(), ni_image->getHeight(),
      reinterpret_cast<uint8_t*>(&image.begin<uint8_t>()[0]));

  // Convert the RGB image to grayscale
  cv::Mat image_gray;
  cv::cvtColor(image, image_gray, CV_BGR2GRAY);

  // Copy the depth image to a float Mat
  cv::Mat_<float> depth_image(ni_depth_image->getHeight(), ni_depth_image->getWidth());
  ni_depth_image->fillDepthImage(ni_depth_image->getWidth(), ni_depth_image->getHeight(), &depth_image.begin()[0]);

  std::vector<cv::Point3f> next_points_3D;
  if(prev_points.size() > 0)
  {
    // Track the points detected from the last frame
    std::cout << "Tracking " << prev_points.size() << " points" << std::endl;
    std::vector<cv::Point2f> next_points;
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(prev_image_gray, image_gray, prev_points, next_points, status, error, cv::Size(5,5), 5);

    // Project all of the good track points to 3D
    for(size_t i=0; i<next_points.size(); ++i)
    {
      if(!status[i]) continue;
      float next_depth = depth_image.at<float>(next_points[i].y, next_points[i].x);

      if(next_depth == shadow_value) continue;
      if(next_depth == no_sample_value) continue;


      cv::circle(image, prev_points[i], 5, cv::Scalar(0));
      cv::circle(image, next_points[i], 5, cv::Scalar(0));
      cv::line(image, prev_points[i], next_points[i], cv::Scalar(128));
    }

    //// Find the affine transformation
    //cv::Mat affine_transform;
    //std::vector<uchar> outliers;
    //cv::estimateAffine3D(prev_points_3D, next_points_3D, affine_transform, inliers);


  }
  
  imshow("image", image);
  imshow("depth", depth_image/10);
  cv::waitKey(50);

  // Switch the new to the old
  image_gray.copyTo(prev_image_gray);
  prev_points_3D = next_points_3D;

  // Detect trackable features in the next frame
  cv::goodFeaturesToTrack(prev_image_gray, prev_points, 50, 0.01, 5);

}

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
    openni_callback_func =
    boost::bind (openni_callback, _1, _2, _3);

  interface->registerCallback(openni_callback_func);
 
  interface->start(); 

  while(true)
  {
    sleep(1);
  }

  return 0;
}
