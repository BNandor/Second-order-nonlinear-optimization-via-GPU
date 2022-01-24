//
// Created by spaceman on 2022. 01. 23..
//

#ifndef PARALLELLBFGS_F1_CUH
#define PARALLELLBFGS_F1_CUH

#include "Problem.cuh"
#include "../AD/function/DFunction.cuh"
#include "../AD/function/DSquareFunction.cuh"
#include "../AD/DDouble.cuh"

class F1 : public Problem {
public:

    __device__ __host__
    DDouble *costFunction(DDouble *parameters, unsigned size) override {
        DDouble *x0 = &parameters[0];
        DDouble *x1 = &parameters[1];
        DDouble *hundred = &parameters[2];
        DDouble *one = &parameters[3];
        DDouble *minOne = &parameters[4];
        parameters[5] = x0->square();
        parameters[6] = *x1 - parameters[5];
        parameters[7] = parameters[6].square();
        parameters[8] = *hundred * parameters[7];

        parameters[9] = *x0 * *minOne;
        parameters[10] = parameters[9] + *one;
        parameters[11] = parameters[10].square();
        parameters[12] = parameters[8] + parameters[11];
        return &parameters[12];
    }
};

#endif //PARALLELLBFGS_F1_CUH
