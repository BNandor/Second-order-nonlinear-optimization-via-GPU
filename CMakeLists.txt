cmake_minimum_required(VERSION 3.17)
project(ParallelLBFGS CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(ParallelLBFGS main.cu core/AD/DDouble.cuh core/AD/function/DFunction.cuh core/common/Tests.cuh core/AD/function/DIDFunction.cuh core/AD/function/DPlusFunction.cuh core/AD/function/DMinusFunction.cuh core/AD/function/DMultiplicationFunction.cuh core/AD/function/Operations.cuh core/problem/Problem.cuh core/problem/F1.cuh core/AD/function/DSquareFunction.cuh core/problem/PlaneFitting.cuh)

set_target_properties(
        ParallelLBFGS
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ParallelLBFGS PROPERTY CUDA_ARCHITECTURES 60)
