//
// Created by spaceman on 2022. 03. 16..
//

#ifndef PARALLELLBFGS_SNLP3DANCHOR_CUH
#define PARALLELLBFGS_SNLP3DANCHOR_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"
#include <math.h>

class SNLP3DAnchor : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 19;
    static const unsigned ThisParameterSize = 3;
    static const unsigned ThisConstantSize = 5;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    SNLP3DAnchor() {
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
        // ((c(0)-x(0))^2 + (c(1)-x(1))^2 + (c(2)-x(2))^2 - c(4)^2)^2
        initOperatorTreePartially(x, trunc(operatorTree[3].value) * 3, 3, 0);
        operatorTree[8] = operatorTree[0] - operatorTree[5];
        operatorTree[9] = operatorTree[8].square();
        operatorTree[10] = operatorTree[1] - operatorTree[6];
        operatorTree[11] = operatorTree[10].square();
        operatorTree[12] = operatorTree[9] + operatorTree[11];
        operatorTree[13] = operatorTree[2] - operatorTree[7];
        operatorTree[14] = operatorTree[13].square();
        operatorTree[15] = operatorTree[12] + operatorTree[14];
        operatorTree[16] = operatorTree[4].square();
        operatorTree[17] = operatorTree[15] - operatorTree[16];
        operatorTree[18] = operatorTree[17].square();
        return &operatorTree[18];
    }
};

#endif //PARALLELLBFGS_SNLP3DANCHOR_CUH
