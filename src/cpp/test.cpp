#include <string>
#include <cstdlib>
#include <chrono>
#include <boost/filesystem.hpp>
#include "beiguang_aoi.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


int main(){
    cv::Mat src;
    cv::Mat dst;
    src = cv::imread("./images/test_2.png");
    BeiGuangAOI_Defects_Detector obj("config.json");
    obj.detect(src, dst);
    cv::imwrite("./result.jpg", dst);
}