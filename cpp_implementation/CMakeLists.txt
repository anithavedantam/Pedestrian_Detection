cmake_minimum_required(VERSION 3.10)
project(PedestrianDetection)

find_package(OpenCV REQUIRED)

add_executable(PedestrianDetection main.cpp)
target_link_libraries(PedestrianDetection ${OpenCV_LIBS})

