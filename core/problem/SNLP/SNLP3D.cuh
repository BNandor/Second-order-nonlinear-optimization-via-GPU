//
// Created by spaceman on 2022. 03. 16..
//

#ifndef PARALLELLBFGS_SNLP3D_CUH
#define PARALLELLBFGS_SNLP3D_CUH

#include "../Problem.cuh"
#include "../../AD/function/DFunction.cuh"
#include "../../AD/function/DSquareFunction.cuh"
#include "../../AD/DDouble.cuh"
#include <math.h>

class SNLP3D : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 20;
    static const unsigned ThisParameterSize = 6;
    static const unsigned ThisConstantSize = 3;
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};
    unsigned ThisJacobianIndices[ThisParameterSize] = {};

    __device__ __host__
    SNLP3D() {
        operatorTreeSize = ThisOperatorTreeSize;
        parameterSize = ThisParameterSize;
        constantSize = ThisConstantSize;
        operatorTree = ThisOperatorTree;
        jacobianIndices = ThisJacobianIndices;
        initIndex();
    }

    __device__ __host__
    void setConstants(double *constants, unsigned constantsSize) {
        initConst(constants, constantsSize);
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
        // ((x(0)-x(3))^2 + (x(1)-x(4))^2 + (x(2)-x(5))^2 - c(2)^2)^2
//        initOperatorTree(x, xSize);
        initOperatorTreePartially(x, trunc(operatorTree[0].value) * 3, 3, 0);
        initOperatorTreePartially(x, trunc(operatorTree[1].value) * 3, 3, 3);

        operatorTree[9] = operatorTree[3] - operatorTree[6];
        operatorTree[10] = operatorTree[9].square();
        operatorTree[11] = operatorTree[4] - operatorTree[7];
        operatorTree[12] = operatorTree[11].square();
        operatorTree[13] = operatorTree[10] + operatorTree[12];
        operatorTree[14] = operatorTree[5] - operatorTree[8];
        operatorTree[15] = operatorTree[14].square();
        operatorTree[16] = operatorTree[13] + operatorTree[15];
        operatorTree[17] = operatorTree[2].square();
        operatorTree[18] = operatorTree[16] - operatorTree[17];
        operatorTree[19] = operatorTree[18].square();

        return &operatorTree[19];
    }
};

#endif //PARALLELLBFGS_SNLP3D_CUH
