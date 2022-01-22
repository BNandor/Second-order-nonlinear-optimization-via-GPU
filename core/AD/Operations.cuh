//
// Created by spaceman on 2022. 01. 22..
//

#ifndef PARALLELLBFGS_OPERATIONS_CUH
#define PARALLELLBFGS_OPERATIONS_CUH

#include "DMultiplicationFunction.cuh"
#include "DPlusFunction.cuh"

__device__
DMultiplicationFunction operator*(DFunction &a, DFunction &b) {
    return DMultiplicationFunction(&a, &b);
}

__device__
DPlusFunction operator+(DFunction &a, DFunction &b) {
    return DPlusFunction(&a, &b);
}

__device__
DMinusFunction operator-(DFunction &a, DFunction &b) {
    return DMinusFunction(&a, &b);
}

#endif //PARALLELLBFGS_OPERATIONS_CUH
