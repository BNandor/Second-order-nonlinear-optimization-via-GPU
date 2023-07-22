//
// Created by spaceman on 2023. 07. 04..
//

#ifndef PARALLELLBFGS_DIXONPRICE2_CUH
#define PARALLELLBFGS_DIXONPRICE2_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class DixonPrice2 : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 10;
    static const unsigned ThisParameterSize = 2;
    static const unsigned ThisConstantSize = 3;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    DixonPrice2() {
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
        //c(0)*((c(2)*(x(0))^2-x(1))^2)
        initOperatorTreePartially(x, trunc(operatorTree[0].value)-1, 1, 0);
        initOperatorTreePartially(x, trunc(operatorTree[1].value), 1, 1);
        operatorTree[5]= operatorTree[3].square();
        operatorTree[6]= operatorTree[2] * operatorTree[5];
        operatorTree[7]= operatorTree[6] - operatorTree[4];
        operatorTree[8]= operatorTree[7].square();
        operatorTree[9]= operatorTree[0] * operatorTree[8];
        return &operatorTree[9];
    }
};

#endif //PARALLELLBFGS_DIXONPRICE2_CUH
