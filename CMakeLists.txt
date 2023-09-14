# cmake_minimum_required(VERSION 3.0.0)
cmake_minimum_required(VERSION 2.7.0...3.27.1)
project(calib_prepro VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.8.0_1")

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(yaml-cpp REQUIRED)
pkg_check_modules(yaml_cpp REQUIRED IMPORTED_TARGET yaml-cpp)

add_definitions(-DCONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config")

# include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}
    main.cpp
)

# link libraries
target_link_libraries( ${PROJECT_NAME}
    ${OpenCV_LIBRARIES}
    PkgConfig::yaml_cpp
)