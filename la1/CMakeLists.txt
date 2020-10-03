cmake_minimum_required(VERSION 3.10)

project(MultilayerPerceptron VERSION 1.0)
include_directories(include)

add_executable(LabAssignment1 src/la1.cpp src/mlp.cpp src/neuron.cpp src/layer.cpp)