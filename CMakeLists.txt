cmake_minimum_required(VERSION 3.17)
project(ParallelLBFGS CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ParallelLBFGS main.cu core/AD/DFloat.cpp core/AD/DFloat.cuh core/AD/DFunction.cpp core/AD/DFunction.cuh core/AD/HeapSet.cuh core/common/Tests.cuh)

set_target_properties(
        ParallelLBFGS
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)