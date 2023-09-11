#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <numeric>
#include <boost/json.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "beiguang_aoi.hpp"

typedef std::array<double, 2> array2d;

const double ROI_RATIO_LOW_BOUND = 0.40;
const double ROI_RATIO_UP_BOUND = 0.95;

// Params to detect ROI
#define FINE_ADAPT_KSIZE_KEY "fine_adapt_ksize"
#define FINE_ADAPT_KSIZE_VAL 99
#define INTERNAL_SHRINKAGE_KEY "internal_shrinkage"
#define INTERNAL_SHRINKAGE_VAL 30
// Params to detect light leaking 
#define LIGHT_LEAKING_THRESH_KEY "light_leaking_thresh"
#define LIGHT_LEAKING_THRESH_VAL 210
// Params to detect dark defects
#define ADAPT_CONST_DARK_KEY "adapt_const_dark"
#define ADAPT_CONST_DARK_VAL 13.0
// Params to detect bright defects
#define ADAPT_CONST_BRIGHT_KEY "adapt_const_bright"
#define ADAPT_CONST_BRIGHT_VAL -9.0
// Params to filter defects (for both dark and bright defects)
#define MIN_DEFECT_AREA_KEY "min_defect_area"
#define MIN_DEFECT_AREA_VAL 6
#define MIN_DEFECT_RECT_WIDTH_KEY "min_defect_rect_width"
#define MIN_DEFECT_RECT_WIDTH_VAL 4
#define MIN_DEFECT_RECT_HEIGHT_KEY "min_defect_rect_height"
#define MIN_DEFECT_RECT_HEIGHT_VAL 4
// Params to calculate uniformity of brightness
#define UNIFORMITY_INTERNAL_SHRINKAGE_KEY "uniformity_internal_shrinkage"
#define UNIFORMITY_INTERNAL_SHRINKAGE_VAL 190
#define UNIFORMITY_CIRCLE_RADIUS_KEY "uniformity_circle_radius"
#define UNIFORMITY_CIRCLE_RADIUS_VAL 33
#define UNIFORMITY_5PTS_THRESH_KEY "uniformity_5pts_thresh"
#define UNIFORMITY_5PTS_THRESH_VAL 0.8
#define UNIFORMITY_13PTS_THRESH_KEY "uniformity_13pts_thresh"
#define UNIFORMITY_13PTS_THRESH_VAL 0.6

BeiGuangAOI_Defects_Detector::BeiGuangAOI_Defects_Detector(const std::string& config_file)
    : m_config_file(config_file)
{
    namespace fs = boost::filesystem;
    namespace json = boost::json;
    const json::kind t_int64 = json::kind::int64;
    const json::kind t_double = json::kind::double_;

    // Initialize the default values
    m_light_leaking_thresh = LIGHT_LEAKING_THRESH_VAL;
    m_fine_adapt_ksize = FINE_ADAPT_KSIZE_VAL;
    m_internal_shrinkage = INTERNAL_SHRINKAGE_VAL;
    m_adapt_const_dark = ADAPT_CONST_DARK_VAL;
    m_adapt_const_bright = ADAPT_CONST_BRIGHT_VAL;
    m_min_defect_area = MIN_DEFECT_AREA_VAL;
    m_min_defect_rect_width = MIN_DEFECT_RECT_WIDTH_VAL;
    m_min_defect_rect_height = MIN_DEFECT_RECT_HEIGHT_VAL;
    m_uniformity_internal_shrinkage = UNIFORMITY_INTERNAL_SHRINKAGE_VAL;
    m_uniformity_circle_radius = UNIFORMITY_CIRCLE_RADIUS_VAL;
    m_uniformity_5pts_thresh = UNIFORMITY_5PTS_THRESH_VAL;
    m_uniformity_13pts_thresh = UNIFORMITY_13PTS_THRESH_VAL;

    if (fs::is_regular_file(fs::path(m_config_file)))
    {   
        std::ifstream ifs(m_config_file);
        std::string input(std::istreambuf_iterator<char>(ifs), {});
        const json::value jv = json::parse(input);
        if (jv.kind() == json::kind::object)
        {
            const auto& obj = jv.get_object();
            if (!obj.empty())
            {
                for (const auto& iter : obj)
                {
                    const json::string_view key = iter.key();
                    const json::value jvv = iter.value();
                    const json::kind kind = jvv.kind();

                    if (key == LIGHT_LEAKING_THRESH_KEY && kind == t_int64)
                        m_light_leaking_thresh = jvv.get_int64();
                    
                    else if (key == FINE_ADAPT_KSIZE_KEY && kind == t_int64)
                        m_fine_adapt_ksize = jvv.get_int64();

                    else if (key == INTERNAL_SHRINKAGE_KEY && kind == t_int64)
                        m_internal_shrinkage = jvv.get_int64();

                    else if (key == ADAPT_CONST_DARK_KEY && kind == t_double)
                        m_adapt_const_dark = jvv.get_double();

                    else if (key == ADAPT_CONST_BRIGHT_KEY && kind == t_double)
                        m_adapt_const_bright = jvv.get_double();
                    
                    else if (key == MIN_DEFECT_AREA_KEY && kind == t_int64)
                        m_min_defect_area = jvv.get_int64();
                    
                    else if (key == MIN_DEFECT_RECT_WIDTH_KEY && kind == t_int64)
                        m_min_defect_rect_width = jvv.get_int64();
                   
                    else if (key == MIN_DEFECT_RECT_HEIGHT_KEY && kind == t_int64)
                        m_min_defect_rect_height = jvv.get_int64();

                    else if (key == UNIFORMITY_INTERNAL_SHRINKAGE_KEY && kind == t_int64)
                        m_uniformity_internal_shrinkage = jvv.get_int64();
                    
                    else if (key == UNIFORMITY_CIRCLE_RADIUS_KEY && kind == t_int64)
                        m_uniformity_circle_radius = jvv.get_int64();

                    else if (key == UNIFORMITY_5PTS_THRESH_KEY && kind == t_double)
                        m_uniformity_5pts_thresh = jvv.get_double();
                    
                    else if (key == UNIFORMITY_13PTS_THRESH_KEY && kind == t_double)
                        m_uniformity_13pts_thresh = jvv.get_double();
                } // end for
            } // end if obj
        } // end if kind
        std::cout << "reading config..." << std::endl;
    } // end if file

}

BeiGuangAOI_Defects_Detector::~BeiGuangAOI_Defects_Detector()
{
    // Nothing needs to free
}

cv::Mat
BeiGuangAOI_Defects_Detector::detect_roi(const cv::Mat& img_gray)
{
    // Adaptive threshold to find edges
    cv::Mat thresh;
    cv::adaptiveThreshold(img_gray, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY, m_fine_adapt_ksize, 0);
    cv::imwrite("./temp/00.jpg", thresh);

    // Remove pepper-salt noises
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);
    cv::imwrite("./temp/01.jpg", thresh);
    cv::morphologyEx(thresh, thresh, cv::MORPH_CLOSE, kernel);
    cv::imwrite("./temp/02.jpg", thresh);
    // Find ROI contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // cv::Mat img_color = cv::imread("/home/zzq/code/aoi/images/test_1.png");
    // cv::drawContours(img_color, contours, -1, cv::Scalar(0,0,255), 3);
    // cv::imwrite("./temp/03.jpg", img_color);

    std::vector<cv::Point> min_cnt;
    double min_area = std::numeric_limits<double>::infinity();
    for (const auto& cnt : contours)
    {
        double area = cv::contourArea(cnt);
        if (area > 1.0e7)      //大于10000000的最小轮廓
        {
            if (area < min_area)
            {
                min_area = area;
                min_cnt = cnt;
            }
        }
    }

    std::vector<std::vector<cv::Point>> min_cnt_1;
    min_cnt_1.push_back(min_cnt);
    cv::Mat img_color = cv::imread("/home/zzq/code/aoi/images/test_2.png");
    cv::drawContours(img_color, min_cnt_1, -1, cv::Scalar(0,0,255), 3);
    cv::imwrite("./temp/04.jpg", img_color);
    
    // Create ROI mask
    cv::Mat mask_rect;
    if (min_cnt.size())
    {
        mask_rect = cv::Mat::zeros(thresh.size(), CV_8UC1);
        cv::Rect rect = cv::boundingRect(min_cnt);
        cv::rectangle(mask_rect, rect, cv::Scalar(255), -1);
    }
    else
        mask_rect = cv::Mat::ones(thresh.size(), CV_8UC1) * 255;
    cv::bitwise_and(thresh, mask_rect, thresh);
    cv::imwrite("./temp/05.jpg", mask_rect);
    cv::imwrite("./temp/06.jpg", thresh);
    
    // Create ROI hull
    std::vector<std::vector<cv::Point>> contours_hull;
    cv::findContours(thresh, contours_hull, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> contours_filtered;
    for (const auto& cnt : contours_hull)
    {
        double area = cv::contourArea(cnt);
        if (area > (2 << 11))  //area > 4096
            contours_filtered.push_back(cnt);
    }
    std::vector<cv::Point> contours_stacked, hull;
    for (const auto& cnt : contours_filtered)
        contours_stacked.insert(contours_stacked.end(), cnt.begin(), cnt.end());
    if (!contours_stacked.size()) return cv::Mat();
    cv::convexHull(contours_stacked, hull);        //凸包检测

    // Create mask of ROI hull
    cv::Mat mask_hull = cv::Mat::zeros(thresh.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> vec_of_hull;
    vec_of_hull.push_back(hull);
    cv::drawContours(mask_hull, vec_of_hull, -1, cv::Scalar(255), -1);
    
    // Create finer-grained mask
    cv::Mat mask_fine = cv::Mat::zeros(thresh.size(), CV_8UC1);
    cv::drawContours(mask_fine, contours_filtered, -1, cv::Scalar(255), -1);
    cv::bitwise_and(mask_fine, mask_hull, mask_fine);

    return mask_fine;
}

cv::Rect 
BeiGuangAOI_Defects_Detector::detect_light_leaking(const cv::Mat& mask_roi, 
    const cv::Mat& img_roi, const cv::Mat& img_in)
{   
    //查看输入
    cv::imwrite("./temp/27.png", mask_roi);
    cv::imwrite("./temp/28.png", img_roi);
    cv::imwrite("./temp/29.png", img_in);

    // Find ROI edges
    cv::Mat edges;
    cv::Canny(mask_roi, edges, 100, 200);
    cv::imwrite("./temp/30.png", edges);
    // Find horizontal edges
    cv::Mat kernel_h = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 1));
    cv::Mat edges_h;
    cv::morphologyEx(edges, edges_h, cv::MORPH_OPEN, kernel_h);
    cv::imwrite("./temp/31.png", edges_h);
    // Find vertical edges
    cv::Mat kernel_v = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 7));
    cv::Mat edges_v;
    cv::morphologyEx(edges, edges_v, cv::MORPH_OPEN, kernel_v);
    cv::imwrite("./temp/32.png", edges_v);

    // Top half, at the horizontal direction 
    int w = mask_roi.size().width;
    int h = mask_roi.size().height;
    cv::Mat edges_top_half = edges_h(cv::Rect(0, 0, w, h / 2));
    cv::imwrite("./temp/33.png", edges_top_half);
    std::vector<std::vector<cv::Point>> cnts_top_half;
    cv::findContours(edges_top_half, cnts_top_half, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> cnts_top_half_stacked;
    for (const auto& cnt : cnts_top_half)
        cnts_top_half_stacked.insert(cnts_top_half_stacked.end(), cnt.begin(), cnt.end());
    int avg_hy = int(cv::mean(cnts_top_half_stacked)[1]);       //图像的上半部分
    // for (auto& i : cnts_top_half_stacked)
    //     std::cout << i.x << "  " << i.y << std::endl;
    std::cout << "avg_hy: " << avg_hy << std::endl;
    std::cout << "top: " << cnts_top_half_stacked.size() << std::endl;

    // Top half, at the vertical direction
    // Left  左上角
    cv::Mat edges_top_left = edges_v(cv::Rect(0, 0, w / 2, h / 2));
    cv::imwrite("./temp/34.png", edges_top_left);
    std::vector<std::vector<cv::Point>> cnts_top_left;
    cv::findContours(edges_top_left, cnts_top_left, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> cnts_top_left_stacked;
    for (const auto& cnt : cnts_top_left)
        cnts_top_left_stacked.insert(cnts_top_left_stacked.end(), cnt.begin(), cnt.end());
    int avg_vx1 = int(cv::mean(cnts_top_left_stacked)[0]);
    std::cout <<  "left: " << cnts_top_half_stacked.size() << std::endl;
    // Right  右上角
    cv::Mat edges_top_right = edges_v(cv::Rect(w - w / 2, 0, w / 2, h / 2));
    cv::imwrite("./temp/35.png", edges_top_right);
    std::vector<std::vector<cv::Point>> cnts_top_right;
    cv::findContours(edges_top_right, cnts_top_right, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> cnts_top_right_stacked;
    for (const auto& cnt : cnts_top_right)
        cnts_top_right_stacked.insert(cnts_top_right_stacked.end(), cnt.begin(), cnt.end());
    int avg_vx2 = int(cv::mean(cnts_top_right_stacked)[0] + w / 2);
    std::cout << "right: " << cnts_top_half_stacked.size() << std::endl;

    // Upward offset
    int offset_y1 = 20;
    // Downward offset
    int offset_y2 = 50;
    int offset_y = (avg_hy > offset_y1 ? avg_hy : 0);
    cv::Mat tmp;
    cv::cvtColor(img_in, tmp, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(tmp, tmp, cv::Size(11, 11), 0.0);
    cv::createCLAHE(2.0, cv::Size(8, 8))->apply(tmp, tmp);
    cv::threshold(tmp, tmp, m_light_leaking_thresh, 255, cv::THRESH_BINARY);
    cv::Mat roi = tmp(cv::Rect(cv::Point(0, offset_y), 
                               cv::Point(w-1, avg_hy+offset_y2))).clone();
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(roi, roi, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(roi, roi, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 3);
    cv::imwrite("./temp/36.png", roi);
    
    std::vector<std::vector<cv::Point>> cnts;
    cv::findContours(roi, cnts, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    int count = 0;
    for (const auto& cnt : cnts)
    {
        double area = cv::contourArea(cnt);
        if (area >= 25.0) count += 1;
    }
    
    cv::Rect defect;
    if (count >= 2)
    {
        defect.x = int(avg_vx1);
        defect.y = int(offset_y);
        defect.width = int(avg_vx2 - avg_vx1);
        defect.height = 400;
    }
    std::cout << "count: " << count << std::endl;
    return defect;
}

UniformityData
BeiGuangAOI_Defects_Detector::detect_brightness_uniformity(const cv::Mat& img_gray,
    const cv::Mat& mask)
{
    UniformityData result;
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); //此处的mask为腐蚀后的mask
    // std::cout << "contours.size: " << contours.size() << std::endl;
    // Lambda function: get the index of the maximum area of contours
    auto get_max_area_contour_id = [&contours]()->int
    {
        double max_area = 0.0;
        int max_area_contour_id = -1;
        for (int j = 0; j < contours.size(); j++) 
        {
            double new_area = cv::contourArea(contours.at(j));
            if (new_area > max_area) 
            {
                max_area = new_area;
                max_area_contour_id = j;
            }
        }
        return max_area_contour_id;
    };

    int idx = get_max_area_contour_id(); //获取到最大的轮廓的id，在腐蚀后的mask上获得的轮廓
    if (idx < 0) return result;

    std::vector<cv::Point> roi = contours.at(idx);
    cv::Rect roi_rect = cv::boundingRect(roi);
    
    // Lambda function: calculate the mean and stdev of sample area
    auto cal_mean_stdev = [&img_gray](const cv::Mat& mask)->array2d
    {
        std::vector<cv::Point> nonzero_idx;
        cv::findNonZero(mask, nonzero_idx);
        std::vector<uchar> vec;
        for (size_t i = 0; i < nonzero_idx.size(); ++i)
            vec.push_back(img_gray.at<uchar>(nonzero_idx[i]));
        
        double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
        double mean = 0.0, stdev = 0.0;
        if (!vec.empty())
        {
            mean = sum / vec.size();
            double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
            stdev = std::sqrt(sq_sum / vec.size() - mean * mean);
        }
        return array2d{mean, stdev};
    };
    
    // // ----------- debug only -----------
    // auto double_to_string = [](const double& v, int prec = 2)->std::string {
    //     std::ostringstream oss_tmp;
    //     oss_tmp << std::fixed;
    //     oss_tmp << std::setprecision(prec);
    //     oss_tmp << v;
    //     return oss_tmp.str();
    // };
    // cv::Mat tmp_5, tmp_13;
    // cv::cvtColor(img_gray, tmp_5, cv::COLOR_GRAY2BGR);
    // cv::cvtColor(img_gray, tmp_13, cv::COLOR_GRAY2BGR);
    // // ----------- debug only -----------

    cv::Mat mask_samples_5 = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::Mat mask_samples_13 = cv::Mat::zeros(mask.size(), CV_8UC1);
    const int& offset = m_uniformity_internal_shrinkage;    //收缩
    const int& radius = m_uniformity_circle_radius;         //半径
    const int divides = 4;
    int width_per_divide = int(roi_rect.width - 2 * offset) / divides;
    int height_per_divide = int(roi_rect.height - 2 * offset) / divides;

    // cv::Mat img_t = img_gray.clone();
    // cv::rectangle(img_t, roi_rect, cv::Scalar(0), 3);
    // cv::imwrite("./temp/41.jpg", img_t);

    // For calculating uniformity of 5/13 sample areas
    double sum_5  = 0.0  , sum_13 = 0.0;
    double min_5  = 255.0, max_5  = 0.0;
    double min_13 = 255.0, max_13 = 0.0;
    for (int row = 0, count = 0; row < 5; ++row)  //0~4
    {
        for (int col = 0; col < 5; ++col)   //0~4
        {
            if (!((row % 2) ^ (col % 2)))   //在roi上选13个点
            {   
                // int m = ((row % 2) ^ (col % 2));
                // std::cout << "row: " << row << "  " << "col: " << col << "       " << m << std::endl;

                int x = roi_rect.x + offset + width_per_divide * col;
                int y = roi_rect.y + offset + height_per_divide * row;
                cv::Point center(x, y);  //取到每个采样圆的中心点
                cv::Mat mask_tmp = cv::Mat::zeros(mask.size(), CV_8UC1);
                cv::circle(mask_tmp, center, radius, cv::Scalar(255), -1);
                cv::imwrite("./temp/42.jpg", mask_tmp);
                // result.each13[count++] = cal_mean_stdev(mask_tmp);
                array2d res = cal_mean_stdev(mask_tmp);
                // std::cout << "res: " << res[0] << " " << res[1] << std::endl;
                result.each13[count++] = res;    //res：<mean, std>
                sum_13 += res[0];
                if (res[0] < min_13) min_13 = res[0];
                if (res[0] > max_13) max_13 = res[0];

                // // ----------- debug only -----------
                // std::string text0 = "#" + std::to_string(count);
                // std::string text1 = "Mean = " + double_to_string(res[0]);
                // std::string text2 = "Stdev =" + double_to_string(res[1]);
                // cv::putText(tmp_5, text0, cv::Point(x-radius, y+radius+50), cv::FONT_HERSHEY_COMPLEX,
                //     2.0, cv::Scalar(0, 0, 255), 2);
                //     cv::putText(tmp_5, text1, cv::Point(x-radius, y+radius+100), cv::FONT_HERSHEY_COMPLEX,
                //     2.0, cv::Scalar(0, 0, 255), 2);
                // cv::putText(tmp_5, text2, cv::Point(x-radius, y+radius+150), cv::FONT_HERSHEY_COMPLEX,
                //     2.0, cv::Scalar(0, 0, 255), 2);
                // cv::circle(tmp_5, center, radius, cv::Scalar(255, 0, 0), 2);
                // // ----------- debug only -----------

                cv::circle(mask_samples_13, center, radius, cv::Scalar(255), -1);
                cv::imwrite("./temp/43.jpg", mask_samples_13);
                // if ((row % 4 == 0) && (col % 4 == 0) || (row == 2 && col == 2))
                if (((row == 1) && (col == 1 || col == 3)) || 
                    ((row == 3) && (col == 1 || col == 3)) ||
                    ((row == 2) && (col == 2)))
                {
                    cv::circle(mask_samples_5, center, radius, cv::Scalar(255), -1);
                    cv::imwrite("./temp/44.jpg", mask_samples_5);
                    sum_5 += res[0];        //res：<mean, std>
                    if (res[0] < min_5) min_5 = res[0];
                    if (res[0] > max_5) max_5 = res[0];
                }
            }
        }
    }

    // result.whole5 = cal_mean_stdev(mask_samples_5);
    // result.whole13 = cal_mean_stdev(mask_samples_13);
    if (max_5 > LDBL_EPSILON)
    {
        result.whole5 = {sum_5 / 5.0, min_5 / max_5};
        if (result.whole5[1] < m_uniformity_5pts_thresh)
            result.is_ok = false;
    }
    if (max_13 > LDBL_EPSILON)
    {
        result.whole13 = {sum_13 / 13.0, min_13 / max_13};
        if (result.whole13[1] < m_uniformity_13pts_thresh)
            result.is_ok = false;
    }

    // std::string text1 = "Mean of 5-areas: " + double_to_string(result.whole5[0]);
    // std::string text2 = "Mean of 13-areas: " + double_to_string(result.whole13[0]);
    // std::string text3 = "Stdev of 5-areas: " + double_to_string(result.whole5[1]);
    // std::string text4 = "Stdev of 13-areas: " + double_to_string(result.whole13[1]);
    // cv::putText(tmp_5, text1, cv::Point(100, 100), cv::FONT_HERSHEY_COMPLEX,
    //     4.0, cv::Scalar(0, 0, 255), 2);
    // cv::putText(tmp_5, text2, cv::Point(100, 200), cv::FONT_HERSHEY_COMPLEX,
    //     4.0, cv::Scalar(0, 0, 255), 2);
    // cv::putText(tmp_5, text3, cv::Point(100, 300), cv::FONT_HERSHEY_COMPLEX,
    //     4.0, cv::Scalar(0, 0, 255), 2);
    // cv::putText(tmp_5, text4, cv::Point(100, 400), cv::FONT_HERSHEY_COMPLEX,
    //     4.0, cv::Scalar(0, 0, 255), 2);
    // cv::imwrite("tmp.png", tmp_5);
    
    return result;
}

std::pair<std::vector<DefectData>, UniformityData>
BeiGuangAOI_Defects_Detector::detect(const cv::Mat& img_in, cv::Mat& img_out)
{
    std::pair<std::vector<DefectData>, UniformityData> results;
    std::vector<DefectData> res_defects;
    img_in.copyTo(img_out);

    // The following operations are all applied to grayscale data,
    // except for drawing defects on color data.
    cv::Mat img_gray;
    cv::cvtColor(img_in, img_gray, cv::COLOR_BGR2GRAY);
    
    // Get ROI
    cv::Mat mask = detect_roi(img_gray);
    cv::imwrite("./temp/50.png", mask);
    if (mask.empty()) return results;
    cv::Mat img_roi;
    cv::bitwise_and(img_gray, mask, img_roi);
    cv::imwrite("./temp/51.png", img_roi);

    // Draw ROI edges on color data
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(img_out, contours, -1, cv::Scalar(255, 0, 0), 1);
    cv::imwrite("./temp/52.png", img_out);

    // Shrink ROI
    cv::Mat mask_eroded = mask.clone();
    cv::drawContours(mask_eroded, contours, -1, cv::Scalar(0), m_internal_shrinkage);
    cv::imwrite("./temp/53.png", mask_eroded);

    // Draw ROI edges after shrinking on color data
    std::vector<std::vector<cv::Point>> contours_eroded;
    cv::findContours(mask_eroded, contours_eroded, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(img_out, contours_eroded, -1, cv::Scalar(0, 255, 0), 1);
    cv::imwrite("./temp/54.png", img_out);

    // Detect uniformity of brightness
    UniformityData res_uniformity = detect_brightness_uniformity(img_gray, mask_eroded);

    // ------------------- Detect defects -------------------
    // Detect dark defects
    cv::Mat thresh;
    cv::adaptiveThreshold(img_roi, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY, 49, m_adapt_const_dark);
    cv::imwrite("./temp/55.png", thresh);
    thresh = ~thresh;
    cv::imwrite("./temp/56.png", thresh);
    cv::bitwise_and(thresh, mask_eroded, thresh);
    cv::imwrite("./temp/57.png", thresh);

    // Filter dark defects
    std::vector<std::vector<cv::Point>> contours_results;
    cv::findContours(thresh, contours_results, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& cnt : contours_results)
    {
        cv::RotatedRect rrect = cv::minAreaRect(cnt);   //旋转矩形（中心点，宽高，角度）
        double area = cv::contourArea(cnt);
        if (area <= 1.0e6 && (rrect.size.width >= m_min_defect_rect_width ||
            rrect.size.height >= m_min_defect_rect_height))
        {
            double cx, cy, rw, rh, phi, ra;
            cx  = rrect.center.x;
            cy  = rrect.center.y;
            rw  = rrect.size.width;
            rh  = rrect.size.height;
            phi = rrect.angle;
            ra  = rw * rh;
            DefectData defect_data(cx, cy, rw, rh, phi, ra, false);
            res_defects.push_back(defect_data);

            cv::Point2f box_pts[4];
            rrect.points(box_pts);
            for (int j = 0; j < 4; j++)
                cv::line(img_out, box_pts[j], box_pts[(j+1)%4], cv::Scalar(0, 0, 255), 1);
        }
    }
    cv::imwrite("./temp/58.png", img_out);


    // Detect bright defects
    cv::adaptiveThreshold(img_roi, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY, 29, m_adapt_const_bright);
    cv::bitwise_and(thresh, mask_eroded, thresh);

    // Filter bright defects
    std::vector<std::vector<cv::Point>> contours_results_b;
    cv::findContours(thresh, contours_results_b, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& cnt : contours_results_b)
    {
        cv::RotatedRect rrect = cv::minAreaRect(cnt);
        double area = cv::contourArea(cnt);
        if (area <= 1.0e6 && (rrect.size.width >= m_min_defect_rect_width ||
            rrect.size.height >= m_min_defect_rect_height))
        {
            double cx, cy, rw, rh, phi, ra;
            cx  = rrect.center.x;
            cy  = rrect.center.y;
            rw  = rrect.size.width;
            rh  = rrect.size.height;
            phi = rrect.angle;
            ra  = rw * rh;
            DefectData defect_data(cx, cy, rw, rh, phi, ra, true);
            res_defects.push_back(defect_data);

            cv::Point2f box_pts[4];
            rrect.points(box_pts);
            for (int j = 0; j < 4; j++)
                cv::line(img_out, box_pts[j], box_pts[(j+1)%4], cv::Scalar(0, 255, 255), 1);
        }
    }

    // Detect light leaking at top
    cv::Rect res_rect = detect_light_leaking(mask, img_roi, img_in);
    if (!res_rect.empty())
    {
        int x = res_rect.x;
        int y = res_rect.y;
        int w = res_rect.width;
        int h = res_rect.height;

        DefectData defect;
        defect.cx = x + w / 2;
        defect.cy = y + h / 2;
        defect.rw = w;
        defect.rh = h;
        defect.phi = 0;
        defect.ra = w * h;

        res_defects.push_back(defect);
        cv::rectangle(img_out, res_rect, cv::Scalar(0, 0, 255), 1);
    }
    
    if (res_defects.size())
    {
        std::string text = "#defects: " + std::to_string(res_defects.size());
        cv::putText(img_out, text, cv::Point(200, 200), cv::FONT_HERSHEY_COMPLEX,
            6.0, cv::Scalar(0, 0, 255), 3);
    }

    results.first = res_defects;
    results.second = res_uniformity;

    return results;
}