//
// Created by spaceman on 2023. 07. 04..
//

#ifndef PARALLELLBFGS_LEVY3_CUH
#define PARALLELLBFGS_LEVY3_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class Levy3 : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 15;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 4;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Levy3() {
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
        // c(0) xindex, c(1) 1, c(2) 4, c(3) 2pi, , c(4)=w = c(1)+(x(0)-c(1))/c(2),
        // (w - c(1))^2*(c(1) + (sin(c(2)*w))^2)
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[5]= operatorTree[4] - operatorTree[1];
        operatorTree[6]= operatorTree[5] / operatorTree[2];
        operatorTree[7]= operatorTree[1] + operatorTree[6];//w
        operatorTree[8]= operatorTree[7] - operatorTree[1];
        operatorTree[9]= operatorTree[8].square();
        operatorTree[10]= operatorTree[3] * operatorTree[7];
        operatorTree[11]= operatorTree[10].sin();
        operatorTree[12]= operatorTree[11].square();
        operatorTree[13]= operatorTree[1] + operatorTree[12];
        operatorTree[14]= operatorTree[9] * operatorTree[13];
        return &operatorTree[14];
    }
};

#endif //PARALLELLBFGS_LEVY3_CUH
