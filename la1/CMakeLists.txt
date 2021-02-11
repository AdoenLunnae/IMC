cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
project(MultilayerPerceptronRegression VERSION 1.0)

include_directories(imc/include)

add_executable(la1 imc/util.cpp imc/MultilayerPerceptron.cpp la1.cpp)
