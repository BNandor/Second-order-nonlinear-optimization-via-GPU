//
// Created by spaceman on 2025. 12. 07..
//

#ifndef PARALLELLBFGS_SHIFTEDSCHWEFEL223_CUH
#define PARALLELLBFGS_SHIFTEDSCHWEFEL223_CUH


#include "../../../Problem.cuh"
#include "../../../../AD/function/DFunction.cuh"
#include "../../../../AD/function/DSquareFunction.cuh"
#include "../../../../AD/DDouble.cuh"

class ShiftedSchwefel223 : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 10;
    static const unsigned ThisParameterSize = 1;
    static const unsigned ThisConstantSize = 2;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    ShiftedSchwefel223() {
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
        //(x(0)+c(1))^10
        initOperatorTreePartially(x, trunc(operatorTree[0].value), 1, 0);
        operatorTree[3]= operatorTree[2] + operatorTree[1];
        operatorTree[4]= operatorTree[3].square();
        operatorTree[5]= operatorTree[4].square();
        operatorTree[6]= operatorTree[5].square();
        operatorTree[7]= operatorTree[2] + operatorTree[1];
        operatorTree[8]= operatorTree[7].square();
        operatorTree[9]= operatorTree[6] * operatorTree[8];
        return &operatorTree[9];
    }
};

#endif //PARALLELLBFGS_SHIFTEDSCHWEFEL223_CUH
