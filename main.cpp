#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>

void cloud_callback(pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr const & cloud, pcl::visualization::CloudViewer * viewer)
{
  std::cout << "Got Cloud" << std::endl;
  if(!viewer->wasStopped())
    viewer->showCloud(cloud);
}

void image_callback(boost::shared_ptr<openni_wrapper::Image> const & image)
{
  std::cout << "Got Image" << std::endl;
}

int main(int argc, char** argv)
{  
  pcl::Grabber * interface = new pcl::OpenNIGrabber();

  pcl::visualization::CloudViewer viewer("viewer");

  boost::function<void (pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr const&)> cloud_callback_func =
         boost::bind (cloud_callback, _1, &viewer);
  interface->registerCallback(cloud_callback_func);

  boost::function<void (boost::shared_ptr<openni_wrapper::Image> const &)> image_callback_func =
         boost::bind (image_callback, _1);
  interface->registerCallback(image_callback_func);

  interface->start();

  while(!viewer.wasStopped())
    sleep(1);
  interface->stop();

  return 0;
}
