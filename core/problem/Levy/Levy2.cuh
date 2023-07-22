//
// Created by spaceman on 2023. 07. 04..
//

#ifndef PARALLELLBFGS_LEVY2_CUH
#define PARALLELLBFGS_LEVY2_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class Levy2 : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 18;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 5;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Levy2() {
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
        // c(0) xindex, c(1) 1, c(2) 4, c(3) pi, c(4) 10, w = c(1)+(x(0)-c(1))/c(2),
        // (w - c(1))^2*(c(1) + c(4)*(sin(c(3)*w+c(1)))^2)
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[6]= operatorTree[5] - operatorTree[1];
        operatorTree[7]= operatorTree[6] / operatorTree[2];
        operatorTree[8]= operatorTree[1] + operatorTree[7];//w
        operatorTree[9]= operatorTree[8] - operatorTree[1];
        operatorTree[10]= operatorTree[9].square();
        operatorTree[11]= operatorTree[3] * operatorTree[8];
        operatorTree[12]= operatorTree[11] + operatorTree[1];
        operatorTree[13]= operatorTree[12].sin();
        operatorTree[14]= operatorTree[13].square();
        operatorTree[15]= operatorTree[4] * operatorTree[14];
        operatorTree[16]= operatorTree[1] + operatorTree[15];
        operatorTree[17]= operatorTree[10] * operatorTree[16];
        return &operatorTree[17];
    }
};

#endif //PARALLELLBFGS_LEVY2_CUH
