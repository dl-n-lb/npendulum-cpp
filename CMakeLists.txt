cmake_minimum_required(VERSION 3.22)

project(n_pendulum)

set(CMAKE_CXX_COMPILER "/usr/bin/g++") # remove this or change if your compiler is different
set(CMAKE_CXX_STANDARD 20)

include_directories(common_cxx/)

add_executable(n_pendulum source/main.cxx)
