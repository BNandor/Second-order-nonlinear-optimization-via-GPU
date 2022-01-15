cmake_minimum_required(VERSION 3.17)
project(ParallelLBFGS CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ParallelLBFGS main.cu core/AD/DDouble.cpp core/AD/DDouble.cuh)

set_target_properties(
        ParallelLBFGS
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)