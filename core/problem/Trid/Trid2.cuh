//
// Created by spaceman on 2022. 12. 08..
//

#ifndef PARALLELLBFGS_TRID2_CUH
#define PARALLELLBFGS_TRID2_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class Trid2 : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 7;
    static const unsigned ThisParameterSize = 2;
    static const unsigned ThisConstantSize = 3;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Trid2() {
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
        //(x(0)*x(1))*c(2)
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        initOperatorTreePartially(x, trunc(operatorTree[1].value), 1, 1);
        operatorTree[5]= operatorTree[3] * operatorTree[4];
        operatorTree[6]= operatorTree[5] * operatorTree[2];
        return &operatorTree[6];
    }
};

#endif //PARALLELLBFGS_TRID2_CUH
