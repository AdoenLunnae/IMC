cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(MultilayerPerceptron VERSION 1.0)
include_directories(include)

add_executable(la1 src/la1.cpp src/mlp.cpp src/neuron.cpp src/layer.cpp)
