//
// Created by spaceman on 2022. 03. 03..
//

#ifndef PARALLELLBFGS_SNLPANCHOR_CUH
#define PARALLELLBFGS_SNLPANCHOR_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"
#include <math.h>

class SNLPAnchor : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 14;
    static const unsigned ThisParameterSize = 2;
    static const unsigned ThisConstantSize = 4;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    SNLPAnchor() {
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
        // ((c(0)-x(0))^2 + (c(1)-x(1))^2 - c(3)^2)^2
        initOperatorTreePartially(x, trunc(operatorTree[2].value) * 2, 2, 0);
        operatorTree[6] = operatorTree[0] - operatorTree[4];
        operatorTree[7] = operatorTree[6].square();
        operatorTree[8] = operatorTree[1] - operatorTree[5];
        operatorTree[9] = operatorTree[8].square();
        operatorTree[10] = operatorTree[7] + operatorTree[9];
        operatorTree[11] = operatorTree[3].square();
        operatorTree[12] = operatorTree[10] - operatorTree[11];
        operatorTree[13] = operatorTree[12].square();
        return &operatorTree[13];
    }
};

#endif //PARALLELLBFGS_SNLPANCHOR_CUH
