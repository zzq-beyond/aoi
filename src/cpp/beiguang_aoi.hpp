#ifndef BeiGuangAOI_HPP
#define BeiGuangAOI_HPP

#include <string>
#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

struct DefectData
{
    double cx;      // Coordinate x of rotated rectangle
    double cy;      // Coordinate y of rotated rectangle
    double rw;      // Width of rotated rectangle
    double rh;      // Height of rotated rectangle
    double phi;     // Angle phi of rotated rectangle
    double ra;      // Area (=rw*rh) of rotated rectangle
    bool is_bright; // Bright defect or not

    DefectData(double cx_=0.0, double cy_=0.0,
        double rw_=0.0, double rh_=0.0,
        double phi_=0.0, double ra_=0.0, bool is_bright_=false)
        : cx(cx_), cy(cy_), rw(rw_), rh(rh_), phi(phi_), ra(ra_), 
          is_bright(is_bright_) {}
};

struct UniformityData
{
    std::array<std::array<double, 2>, 13> each13;   // [mean, stdev] of 13 sample areas
    std::array<double, 2> whole5;                   // [mean, uniformity] of 5 sample areas
    std::array<double, 2> whole13;                  // [mean, uniformity] of 13 sample areas
    bool is_ok;                                     // Reach the standard or not

    UniformityData()
    {
        for (size_t i = 0; i < each13.size(); ++i)
        {
            std::array<double, 2> item = {0.0, 0.0};
            each13[i] = item;
        }
        whole5[0] = 0.0;
        whole5[1] = 0.0;
        whole13[0] = 0.0;
        whole13[1] = 0.0;
        is_ok = true;
    }
};

class BeiGuangAOI_Defects_Detector
{
private:
    // Config file of algo
    std::string m_config_file;

    // The following params can be configured in config.json file
    // Params to detect ROI
    int m_fine_adapt_ksize;
    int m_internal_shrinkage;
    // Params to detect light leaking 
    int m_light_leaking_thresh;
    // Params to detect dark defects
    double m_adapt_const_dark;
    // Params to detect bright defects
    double m_adapt_const_bright;
    // Params to filter defects (for both dark and bright defects)
    int m_min_defect_area;
    int m_min_defect_rect_width;
    int m_min_defect_rect_height;
    // Params to calculate uniformity of brightness
    int m_uniformity_internal_shrinkage;
    int m_uniformity_circle_radius;
    double m_uniformity_5pts_thresh;
    double m_uniformity_13pts_thresh;

    cv::Mat detect_roi(const cv::Mat& img_gray);
    cv::Rect detect_light_leaking(const cv::Mat& mask_roi, 
        const cv::Mat& img_roi, const cv::Mat& img_in);
    UniformityData detect_brightness_uniformity(const cv::Mat& img_gray, 
        const cv::Mat& mask);
public:
    BeiGuangAOI_Defects_Detector(const std::string& config_file = "");
    ~BeiGuangAOI_Defects_Detector();
    std::pair<std::vector<DefectData>, UniformityData> detect(
        const cv::Mat& img_in, cv::Mat& img_out);
};

#endif