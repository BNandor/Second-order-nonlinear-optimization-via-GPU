//
// Created by spaceman on 2022. 01. 23..
//

#ifndef PARALLELLBFGS_PROBLEM_CUH
#define PARALLELLBFGS_PROBLEM_CUH

#include "../AD/DDouble.cuh"

class Problem {
public:
    __device__ __host__
    virtual DDouble *costFunction(DDouble *parameters, unsigned size) {
        assert(false);
        return nullptr;
    };
};

#endif //PARALLELLBFGS_PROBLEM_CUH
