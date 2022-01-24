//
// Created by spaceman on 2022. 01. 23..
//

#ifndef PARALLELLBFGS_PROBLEM_CUH
#define PARALLELLBFGS_PROBLEM_CUH

#include "../AD/DDouble.cuh"

class Problem {
public:
    __device__ __host__
    virtual DDouble *eval(double *x, unsigned xSize) {
        assert(false);
        return nullptr;
    };
};

#endif //PARALLELLBFGS_PROBLEM_CUH
