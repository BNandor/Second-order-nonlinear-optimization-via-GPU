//
// Created by spaceman on 2023. 07. 04..
//

#ifndef PARALLELLBFGS_DIXONPRICE_CUH
#define PARALLELLBFGS_DIXONPRICE_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class DixonPrice1 : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 5;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 2;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    DixonPrice1() {
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
        //(x(0)-c(1))^2
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[3]= operatorTree[2] - operatorTree[1];
        operatorTree[4]= operatorTree[3].square();
        return &operatorTree[4];
    }
};

#endif //PARALLELLBFGS_DIXONPRICE_CUH
