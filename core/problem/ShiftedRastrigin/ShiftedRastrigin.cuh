//
// Created by spaceman on 2025. 12. 07..
//

#ifndef PARALLELLBFGS_SHIFTEDRASTRIGIN_CUH
#define PARALLELLBFGS_SHIFTEDRASTRIGIN_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class ShiftedRastrigin : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 13;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 4;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    ShiftedRastrigin() {
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
        //(x(0)+c(3))^2 - c(1)*cos(c(2)*(x(0)+c(3)))+ c(1)
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[5]= operatorTree[4] + operatorTree[3];
        operatorTree[6]= operatorTree[5].square();
        operatorTree[7]= operatorTree[4] + operatorTree[3];
        operatorTree[8]= operatorTree[2] * operatorTree[7];
        operatorTree[9]= operatorTree[8].cos();
        operatorTree[10]= operatorTree[1] * operatorTree[9];
        operatorTree[11]= operatorTree[6] - operatorTree[10];
        operatorTree[12]= operatorTree[11] + operatorTree[1];
        return &operatorTree[12];
    }
};

#endif //PARALLELLBFGS_SHIFTEDRASTRIGIN_CUH

