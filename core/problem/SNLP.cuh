//
// Created by spaceman on 2022. 02. 19..
//

#ifndef PARALLELLBFGS_SNLP_CUH
#define PARALLELLBFGS_SNLP_CUH

#include "Problem.cuh"
#include "../AD/function/DFunction.cuh"
#include "../AD/function/DSquareFunction.cuh"
#include "../AD/DDouble.cuh"
#include <math.h>

class SNLP : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 15;
    static const unsigned ThisParameterSize = 4;
    static const unsigned ThisConstantSize = 3;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    SNLP() {
        operatorTreeSize = ThisOperatorTreeSize;
        parameterSize = ThisParameterSize;
        constantSize = ThisConstantSize;
        operatorTree = ThisOperatorTree;
        jacobianIndices = ThisJacobianIndices;
        initIndex();
    }

    __device__ __host__
    void setConstants(double *constants, unsigned constantsSize) {
        initConst(constants, constantsSize);
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
        // ((x(0)-x(2))^2 + (x(1)-x(3))^2 - c(2)^2)^2
//        initOperatorTree(x, xSize);
        initOperatorTreePartially(x, trunc(operatorTree[0].value) * 2, 2, 0);
        initOperatorTreePartially(x, trunc(operatorTree[1].value) * 2, 2, 2);
        operatorTree[7] = operatorTree[3] - operatorTree[5];
        operatorTree[8] = operatorTree[7].square();
        operatorTree[9] = operatorTree[4] - operatorTree[6];
        operatorTree[10] = operatorTree[9].square();
        operatorTree[11] = operatorTree[8] + operatorTree[10];
        operatorTree[12] = operatorTree[11].sqrt();
        operatorTree[13] = operatorTree[2] - operatorTree[12];
        operatorTree[14] = operatorTree[13].square();
        return &operatorTree[14];
    }
};

#endif //PARALLELLBFGS_SNLP_CUH
