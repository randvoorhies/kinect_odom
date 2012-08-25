kinect_odom
===========

Odometry from a Kinect sensor

This is a project to play around with visual odometry using a kinect.

Current Status
=============

The basic functionality is not in place, and the algorithm produces reasonable looking velocity measurements.

The algorithm (for now) is as follows:
1. Detect features in the previous frame
1. Track those features using Lucas & Kanade Optical Flow to get a corresponding set of features in the current frame
1. Project both feature sets into 3D using the kinect depth map and a pinhole camera model
1. Use the Iterative Closest Point algorithm to align the two 3D point clouds
1. Repeat!

Future Work
==========

The end goal is to mimic the algorithm found in "Real-Time Stereo Visual
Odometry for Autonomous Ground Vehicles" (Howard, 2008). This will require:

1. Use FAST to detect features
1. Match features using an n^2 SAD matcher, instead of LK tracking
1. Use Levenberg-Marquardt to solve the alignment, instead of ICP
