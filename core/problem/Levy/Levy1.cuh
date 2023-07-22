//
// Created by spaceman on 2023. 07. 04..
//

#ifndef PARALLELLBFGS_LEVY1_CUH
#define PARALLELLBFGS_LEVY1_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class Levy1 : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 11;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 4;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Levy1() {
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
        //c(0) xindex, c(1) 1, c(2) 4, c(3) pi, w = c(1)+(x(0)-c(1))/c(2),
        // sin(c(3)*w)^2
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[5]= operatorTree[4] - operatorTree[1];
        operatorTree[6]= operatorTree[5] / operatorTree[2];
        operatorTree[7]= operatorTree[1] + operatorTree[6];//w
        operatorTree[8]= operatorTree[3] * operatorTree[7];
        operatorTree[9]= operatorTree[8].sin();
        operatorTree[10]= operatorTree[9].square();
        return &operatorTree[10];
    }
};

#endif //PARALLELLBFGS_LEVY1_CUH
