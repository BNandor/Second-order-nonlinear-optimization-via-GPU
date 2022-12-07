//
// Created by spaceman on 2022. 12. 03..
//

#ifndef PARALLELLBFGS_ROSENBROCK_CUH
#define PARALLELLBFGS_ROSENBROCK_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class Rosenbrock : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 13;
    static const unsigned ThisParameterSize = 2;
    static const unsigned ThisConstantSize = 4;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Rosenbrock() {
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
        //(c(2) - x(0))^2 + c(3)*(x(1) - x(0)^2)^2
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        initOperatorTreePartially(x, trunc(operatorTree[1].value), 1, 1);
        operatorTree[6]= operatorTree[2] - operatorTree[4];
        operatorTree[7]= operatorTree[6].square();
        operatorTree[8]= operatorTree[4].square();
        operatorTree[9]= operatorTree[5] - operatorTree[8];
        operatorTree[10]= operatorTree[9].square();
        operatorTree[11]= operatorTree[3] * operatorTree[10];
        operatorTree[12]= operatorTree[7] + operatorTree[11];
        return &operatorTree[12];
    }
};
#endif //PARALLELLBFGS_ROSENBROCK_CUH
