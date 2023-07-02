//
// Created by spaceman on 2023. 07. 02..
//

#ifndef PARALLELLBFGS_MICHALEWICZ_CUH
#define PARALLELLBFGS_MICHALEWICZ_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"

class Michalewicz : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 20;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 3;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Michalewicz() {
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
        //c(1)*sin(x(0))*(((((sin(c(2)*(x(0)^2)))^2)^2)^2)^2)*(((sin(c(2)*(x(0)^2)))^2)^2)
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[4]= operatorTree[3].sin();
        operatorTree[5]= operatorTree[1] * operatorTree[4];
        operatorTree[6]= operatorTree[3].square();
        operatorTree[7]= operatorTree[2] * operatorTree[6];
        operatorTree[8]= operatorTree[7].sin();
        operatorTree[9]= operatorTree[8].square();
        operatorTree[10]= operatorTree[9].square();
        operatorTree[11]= operatorTree[10].square();
        operatorTree[12]= operatorTree[11].square();
        operatorTree[13]= operatorTree[5] * operatorTree[12];
        operatorTree[14]= operatorTree[3].square();
        operatorTree[15]= operatorTree[2] * operatorTree[14];
        operatorTree[16]= operatorTree[15].sin();
        operatorTree[17]= operatorTree[16].square();
        operatorTree[18]= operatorTree[17].square();
        operatorTree[19]= operatorTree[13] * operatorTree[18];
        return &operatorTree[19];
    }
};

#endif //PARALLELLBFGS_MICHALEWICZ_CUH
