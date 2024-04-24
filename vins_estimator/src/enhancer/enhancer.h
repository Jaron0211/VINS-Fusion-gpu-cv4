#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <numeric>

std::pair<cv::Mat,cv::Mat> adaptive_correction_stereo(cv::Mat image0, cv::Mat image1);
cv::Mat adaptive_correction_mono(cv::Mat image0);