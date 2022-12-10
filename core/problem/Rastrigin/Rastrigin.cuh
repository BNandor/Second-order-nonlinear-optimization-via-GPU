//
// Created by spaceman on 2022. 12. 10..
//

#ifndef PARALLELLBFGS_RASTRIGIN_CUH
#define PARALLELLBFGS_RASTRIGIN_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class Rastrigin : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 10;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 3;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Rastrigin() {
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
        //x(0)^2 - c(1)*cos(c(2)*x(0))+ c(1)
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[4]= operatorTree[3].square();
        operatorTree[5]= operatorTree[2] * operatorTree[3];
        operatorTree[6]= operatorTree[5].cos();
        operatorTree[7]= operatorTree[1] * operatorTree[6];
        operatorTree[8]= operatorTree[4] - operatorTree[7];
        operatorTree[9]= operatorTree[8] + operatorTree[1];
        return &operatorTree[9];
    }
};

#endif //PARALLELLBFGS_RASTRIGIN_CUH
