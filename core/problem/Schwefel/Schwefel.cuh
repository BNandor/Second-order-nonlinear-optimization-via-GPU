//
// Created by spaceman on 2023. 07. 05..
//

#ifndef PARALLELLBFGS_SCHWEFEL_CUH
#define PARALLELLBFGS_SCHWEFEL_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class Schwefel : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 10;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 4;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Schwefel() {
        operatorTreeSize = ThisOperatorTreeSize;
        parameterSize = ThisParameterSize;
        constantSize = ThisConstantSize;
        operatorTree = ThisOperatorTree;
        jacobianIndices = ThisJacobianIndices;
        jacobianIndices = ThisJacobianIndices;
        initIndex();
    }

    __device__ __host__
    void setConstants(double *constants, unsigned constantsSize) {
        initConst(constants, constantsSize);
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
        //  c(3) - x(0)*sin(sqrt(x(0))),  c(0) xindex, c(1)  -1, c(2)  1 c(3) 418.982
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        if(operatorTree[4].value < 0) {
            operatorTree[5] = operatorTree[4] * operatorTree[1];
        }else{
            operatorTree[5] = operatorTree[4] * operatorTree[2];
        }
        operatorTree[6] = operatorTree[5].sqrt();
        operatorTree[7] = operatorTree[6].sin();
        operatorTree[8] = operatorTree[4] * operatorTree[7];
        operatorTree[9] = operatorTree[3] - operatorTree[8];
        return &operatorTree[9];
    }
};

#endif //PARALLELLBFGS_SCHWEFEL_CUH
