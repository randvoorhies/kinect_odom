#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


void openni_callback(
    boost::shared_ptr<openni_wrapper::Image> const & ni_image,
    boost::shared_ptr<openni_wrapper::DepthImage> const & ni_depth_image,
    float constant,
    boost::mutex * mtx)
{
  float const focal_length = 1.0 / constant;
  std::cout << "Got Data - focal length: " << focal_length << std::endl;

  cv::Mat image(ni_image->getHeight(), ni_image->getWidth(), CV_8UC3);
  ni_image->fillRGB( ni_image->getWidth(), ni_image->getHeight(),
      reinterpret_cast<uint8_t*>(&image.begin<uint8_t>()[0]));

  {
    imshow("image", image);
    cv::waitKey(50);
  }
}

int main(int argc, char** argv)
{
  boost::mutex mtx;

  pcl::Grabber * interface = new pcl::OpenNIGrabber();

  boost::function<void (boost::shared_ptr<openni_wrapper::Image> const &, boost::shared_ptr<openni_wrapper::DepthImage> const&, float)> openni_callback_func =
         boost::bind (openni_callback, _1, _2, _3, &mtx);
  interface->registerCallback(openni_callback_func);
 
  interface->start(); 

  while(true)
  {
    sleep(1);
  }

  return 0;
}
