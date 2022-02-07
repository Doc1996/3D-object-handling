cmake_minimum_required(VERSION 3.5)

project(Vision)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

add_executable(${PROJECT_NAME}
    main.cpp)

include("C:/PROGRAMI/PackageManager/vcpkg/scripts/buildsystems/vcpkg.cmake")

# Find Packages
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(realsense2 REQUIRED)
find_package(Eigen3 REQUIRED)

# [C/C++]>[General]>[Additional Include Directories]
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# [C/C++]>[Preprocessor]>[Preprocessor Definitions]
add_definitions(${PCL_DEFINITIONS})

# [Linker]>[General]>[Additional Library Directories]
link_directories(${PCL_LIBRARY_DIRS})

# [Linker]>[Input]>[Additional Dependencies]
target_link_libraries(${PROJECT_NAME}
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${realsense2_LIBRARY}
    Eigen3::Eigen)
