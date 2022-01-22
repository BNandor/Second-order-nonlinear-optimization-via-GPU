cmake_minimum_required(VERSION 3.17)
project(ParallelLBFGS CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ParallelLBFGS main.cu core/AD/DFloat.cuh core/AD/DFunction.cuh core/common/Tests.cuh core/AD/DIDFunction.cuh core/AD/DPlusFunction.cuh core/AD/DMinusFunction.cuh core/AD/DMultiplicationFunction.cuh core/AD/Operations.cuh)

set_target_properties(
        ParallelLBFGS
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)