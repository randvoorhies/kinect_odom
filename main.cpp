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
  static int threshold = 3;
  static cv::Mat prev_image_gray = cv::Mat_<uint8_t>(0,0);
  static std::vector<cv::Point2f> prev_points = std::vector<cv::Point2f>();

  float const focal_length = 1.0 / constant;

  // Copy the image into a cv::Mat
  cv::Mat image(ni_image->getHeight(), ni_image->getWidth(), CV_8UC3);
  ni_image->fillRGB( ni_image->getWidth(), ni_image->getHeight(),
      reinterpret_cast<uint8_t*>(&image.begin<uint8_t>()[0]));

  // Convert the image to grayscale
  cv::Mat image_gray;
  cv::cvtColor(image, image_gray, CV_BGR2GRAY);

  if(prev_points.size() > 0)
  {
    std::cout << "Tracking " << prev_points.size() << " points" << std::endl;
    std::vector<cv::Point2f> next_points;
    std::vector<uchar> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(prev_image_gray, image_gray, prev_points, next_points, status, error, cv::Size(5,5), 5);
  }
  
  imshow("image", image);
  cv::waitKey(50);

  cv::goodFeaturesToTrack(image_gray, prev_points, 50, 0.01, 5);
  std::cout << "Detected " << prev_points.size() << " points" << std::endl;
  image_gray.copyTo(prev_image_gray);
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
