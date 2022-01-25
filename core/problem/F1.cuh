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
    static const unsigned F1operatorTreeSize = 13;
    static const unsigned F1parameterSize = 2;
    static const unsigned F1constantSize = 3;
    double F1J[F1parameterSize];
    DDouble F1operatorTree[F1operatorTreeSize] = {};

    __device__ __host__
    F1(double *constants, unsigned constantsSize) {
        operatorTreeSize = F1operatorTreeSize;
        parameterSize = F1parameterSize;
        constantSize = F1constantSize;
        operatorTree = F1operatorTree;
        J = F1J;
        initConst(constants, constantsSize);
        initIndex();
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
        initOperatorTree(x, xSize);
        DDouble *hundred = &operatorTree[0];
        DDouble *one = &operatorTree[1];
        DDouble *minOne = &operatorTree[2];
        DDouble *x0 = &operatorTree[3];
        DDouble *x1 = &operatorTree[4];

        operatorTree[5] = x0->square();
        operatorTree[6] = *x1 - operatorTree[5];
        operatorTree[7] = operatorTree[6].square();
        operatorTree[8] = *hundred * operatorTree[7];

        operatorTree[9] = *x0 * *minOne;
        operatorTree[10] = operatorTree[9] + *one;
        operatorTree[11] = operatorTree[10].square();
        operatorTree[12] = operatorTree[8] + operatorTree[11];
        return &operatorTree[12];
    }
};

#endif //PARALLELLBFGS_F1_CUH
