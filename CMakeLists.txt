cmake_minimum_required(VERSION 3.7)

set(CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS, "${CXX_FLAGS} ${TORCH_CXX_FLAGS}  -pthread -std")

project(Pytorch-CNN-classifier)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

file(GLOB project_SRCS src/*.cpp)

add_executable(Pytorch-CNN-classifier ${project_SRCS})
target_link_libraries(Pytorch-CNN-classifier ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET Pytorch-CNN-classifier PROPERTY CXX_STANDARD 17)