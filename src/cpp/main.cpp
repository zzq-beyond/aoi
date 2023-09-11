// #include <string>
// #include <cstdlib>
// #include <chrono>
// #include <boost/filesystem.hpp>
// #include "beiguang_aoi.hpp"

// int main(int argc, const char* argv[])
// {
//     namespace bfs = boost::filesystem;

//     // Check input
//     if (argc != 2)
//     {
//         std::cout << "Usage: " << argv[0] << " </path/to/image_file/or/image_folder>\n";
//         std::exit(1);
//     }
//     if (!bfs::exists(argv[1]))
//     {
//         std::cout << "The input path, \"" << argv[1] << "\" does not exist.\n";
//         std::exit(1);
//     }
//     bfs::path image_input_path(argv[1]);

//     // Create the folder to save results
//     bfs::path results_ok_path("./results/ok");
//     bfs::path results_ng_path("./results/ng");
//     if (!bfs::exists(results_ok_path))
//     {
//         bfs::create_directories(results_ok_path);
//     }
//     if (!bfs::exists(results_ng_path))
//     {
//         bfs::create_directories(results_ng_path);
//     }

//     // List all images
//     std::vector<std::string> all_images;

//     if (bfs::is_directory(image_input_path))
//     {
//         for (auto iter = bfs::directory_iterator(image_input_path); 
//              iter != bfs::directory_iterator(); ++iter)
//         {
//             if (bfs::is_regular_file(iter->path()))
//             {
//                 all_images.push_back(iter->path().string());
//             }
//         }
//     }
//     else
//     {
//         all_images.push_back(image_input_path.string());
//     }

//     BeiGuangAOI_Defects_Detector detector("./config.json");

//     // Detect
//     for (const auto& img_f : all_images)
//     {
//         cv::Mat img_in;
//         img_in = cv::imread(img_f);
        
//         auto start = std::chrono::high_resolution_clock::now();

//         cv::Mat img_out;
//         std::pair<std::vector<DefectData>, UniformityData> results;
//         results = detector.detect(img_in, img_out);
//         const std::vector<DefectData>& res_defects = results.first;
//         const UniformityData& res_uniformity = results.second;

//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> dura = end - start;

//         std::string img_name = bfs::path(img_f).filename().string();
//         std::string img_save_path;
//         if (res_defects.size() > 0)
//             img_save_path = results_ng_path.string() + "/" + img_name;
//         else
//             img_save_path = results_ok_path.string() + "/" + img_name;
//         cv::imwrite(img_save_path, img_out);

//         // Print results
//         std::cout << "---------------------------------------" << std::endl;
//         std::cout << "image: " << img_f << std::endl;
//         std::cout << "inference time: " << dura.count() << "ms" << std::endl;
//         std::cout << "count of defects: " << res_defects.size() << std::endl;
//         if (res_defects.size())
//         {
//             std::cout << "coordinate of defects:" << std::endl;
//             for (const auto& res : res_defects)
//             {
//                 std::cout << std::fixed << std::showpoint << std::setprecision(3)
//                           << "\t[cx, cy, rw, rh, phi, ra] = " << "["
//                           << res.cx << ", " << res.cy << ", "
//                           << res.rw << ", " << res.rh << ", "
//                           << res.phi << ", " << res.ra << "]\n";
//             }
//         }
//         std::cout << "brightness unifomity for each area:" << std::endl;
//         for (int i = 0; i < 13; ++i)
//         {
//             std::cout << std::fixed << std::showpoint << std::setprecision(3)
//                       << "\t[mean, stdev](" << i << ") = [" 
//                       << res_uniformity.each13[i][0] << ", "
//                       << res_uniformity.each13[i][1] << "]\n";
//         }
//         std::cout << "brightness unifomity for 5 areas:" << std::endl;
//         std::cout << std::fixed << std::showpoint << std::setprecision(3)
//                   << "\t[mean, uniformity]" << " = [" 
//                   << res_uniformity.whole5[0] << ", "
//                   << res_uniformity.whole5[1] * 100 << "%]\n";
//         std::cout << "brightness unifomity for 13 areas:" << std::endl;
//         std::cout << std::fixed << std::showpoint << std::setprecision(3)
//                   << "\t[mean, uniformity]" << " = [" 
//                   << res_uniformity.whole13[0] << ", "
//                   << res_uniformity.whole13[1] * 100 << "%]\n";
//         std::cout << "brightness unifomity result: " << (res_uniformity.is_ok ? "OK" : "NG") << std::endl;
//         std::cout << std::endl;
//     }

//     return 0;
// }