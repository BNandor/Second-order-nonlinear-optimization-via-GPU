//
// Created by spaceman on 2023. 07. 05..
//

#ifndef PARALLELLBFGS_SUMSQUARES_CUH
#define PARALLELLBFGS_SUMSQUARES_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class SumSquares : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 5;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 2;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    SumSquares() {
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
        //  c(1)*x(0)^2";  c(0) xindex, c(1)  xindex+1
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[3]= operatorTree[2].square();
        operatorTree[4]= operatorTree[1] * operatorTree[3];
        return &operatorTree[4];
    }
};

#endif //PARALLELLBFGS_SUMSQUARES_CUH
