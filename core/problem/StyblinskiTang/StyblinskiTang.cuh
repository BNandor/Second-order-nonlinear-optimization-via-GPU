//
// Created by spaceman on 2022. 12. 07..
//

#ifndef PARALLELLBFGS_STYBLINSKITANG_CUH
#define PARALLELLBFGS_STYBLINSKITANG_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class StyblinskiTang : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 14;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 4;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    StyblinskiTang() {
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
        //((x(0)^2)*(x(0)^2) - c(1)*x(0)^2 + c(2)*x(0))*c(3)
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[5]= operatorTree[4].square();
        operatorTree[6]= operatorTree[4].square();
        operatorTree[7]= operatorTree[5] * operatorTree[6];
        operatorTree[8]= operatorTree[4].square();
        operatorTree[9]= operatorTree[1] * operatorTree[8];
        operatorTree[10]= operatorTree[7] - operatorTree[9];
        operatorTree[11]= operatorTree[2] * operatorTree[4];
        operatorTree[12]= operatorTree[10] + operatorTree[11];
        operatorTree[13]= operatorTree[12] * operatorTree[3];
        return &operatorTree[13];
    }
};
#endif //PARALLELLBFGS_STYBLINSKITANG_CUH
