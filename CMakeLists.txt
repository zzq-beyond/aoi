cmake_minimum_required(VERSION 3.16)

project(beiguang_aoi VERSION 0.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCV 4.4 REQUIRED )
find_package(Boost 1.77 COMPONENTS filesystem json REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

add_library(
    ${PROJECT_NAME} SHARED 
    src/cpp/beiguang_aoi.cpp
)
target_link_libraries(
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

add_executable(
    test_${PROJECT_NAME} 
    src/cpp/main.cpp 
)
target_link_libraries(
    test_${PROJECT_NAME} PRIVATE ${PROJECT_NAME}
    ${OpenCV_LIBS} 
    ${Boost_LIBRARIES}
)
file(
    COPY 
    "./config.json" "./images" 
    DESTINATION ${CMAKE_BINARY_DIR}/bin
)