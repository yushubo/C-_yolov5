cmake_minimum_required(VERSION 3.29)

# 项目名称
project(YOLOv5)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 查找 OpenCV 包
find_package(OpenCV REQUIRED)

# 添加可执行文件
add_executable(yolov5 yolov5.cpp)

# 链接 OpenCV 库
target_link_libraries(yolov5 ${OpenCV_LIBS})

# 设置编译器警告（可选）
if (MSVC)
    target_compile_options(yolov5 PRIVATE /W4)
else()
    target_compile_options(yolov5 PRIVATE -Wall -Wextra -pedantic)
endif()