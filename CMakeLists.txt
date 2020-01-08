cmake_minimum_required(VERSION 3.14)
project(OeSNN)

set(CMAKE_CXX_STANDARD 14)

add_executable(OeSNN main.cpp eSNN.h eSNN.cpp)