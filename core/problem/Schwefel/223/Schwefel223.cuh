//
// Created by spaceman on 2022. 12. 10..
//

#ifndef PARALLELLBFGS_SCHWEFEL223_CUH
#define PARALLELLBFGS_SCHWEFEL223_CUH

#include "../../Problem.cuh"
#include "../../../AD/function/DFunction.cuh"
#include "../../../AD/function/DSquareFunction.cuh"
#include "../../../AD/DDouble.cuh"

class Schwefel223 : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 7;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 1;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    Schwefel223() {
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
        //x(0)^10
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[2]= operatorTree[1].square();
        operatorTree[3]= operatorTree[2].square();
        operatorTree[4]= operatorTree[3].square();
        operatorTree[5]= operatorTree[1].square();
        operatorTree[6]= operatorTree[4] * operatorTree[5];
        return &operatorTree[6];
    }
};
#endif //PARALLELLBFGS_SCHWEFEL223_CUH
