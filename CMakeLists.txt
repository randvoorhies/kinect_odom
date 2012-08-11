cmake_minimum_required(VERSION 2.8)
project(kinect_odom)

set(CMAKE_CXX_COMPILER g++-4.6)

add_executable(kinect_odom main.cpp)

find_package(PkgConfig)

pkg_check_modules(PCL_IO pcl_io-1.5)
include_directories(${PCL_IO_INCLUDE_DIRS})
target_link_libraries(kinect_odom ${PCL_IO_LIBRARIES})

pkg_check_modules(PCL_COMMON pcl_common-1.5)
include_directories(${PCL_COMMON_INCLUDE_DIRS})
target_link_libraries(kinect_odom ${PCL_COMMON_LIBRARIES})

pkg_check_modules(PCL_VISUALIZATION pcl_visualization-1.5)
include_directories(${PCL_VISUALIZATION_INCLUDE_DIRS})
target_link_libraries(kinect_odom ${PCL_VISUALIZATION_LIBRARIES})

pkg_check_modules(OPENNI openni)
include_directories(${OPENNI_INCLUDE_DIRS})
target_link_libraries(kinect_odom ${OPENNI_LIBRARIES})

pkg_check_modules(OPENCV opencv)
include_directories(${OPENCV_INCLUDE_DIRS})
target_link_libraries(kinect_odom ${OPENCV_LIBRARIES})

include_directories(/usr/include/vtk-5.8)

add_definitions(-Wno-deprecated -std=c++0x)

