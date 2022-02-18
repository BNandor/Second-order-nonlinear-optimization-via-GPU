//
// Created by spaceman on 2022. 02. 18..
//

#ifndef PARALLELLBFGS_ROSENBROCK2D_CUH
#define PARALLELLBFGS_ROSENBROCK2D_CUH

#include "Problem.cuh"
#include "../AD/function/DFunction.cuh"
#include "../AD/function/DSquareFunction.cuh"
#include "../AD/DDouble.cuh"

class Rosenbrock2D : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 11;
    static const unsigned ThisParameterSize = 2;
    static const unsigned ThisConstantSize = 2;
    double ThisJ[ThisParameterSize];
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};

    __device__ __host__
    Rosenbrock2D() {
        operatorTreeSize = ThisOperatorTreeSize;
        parameterSize = ThisParameterSize;
        constantSize = ThisConstantSize;
        operatorTree = ThisOperatorTree;
        J = ThisJ;
        initIndex();
    }

    __device__ __host__
    void setConstants(double *constants, unsigned constantsSize) {
        initConst(constants, constantsSize);
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
        //(c(0) - x(0))^2 + c(1)*(x(1) - x(0)^2)^2
        initOperatorTree(x, xSize);
        operatorTree[4] = operatorTree[0] - operatorTree[2];
        operatorTree[5] = operatorTree[4].square();
        operatorTree[6] = operatorTree[2].square();
        operatorTree[7] = operatorTree[3] - operatorTree[6];
        operatorTree[8] = operatorTree[7].square();
        operatorTree[9] = operatorTree[1] * operatorTree[8];
        operatorTree[10] = operatorTree[5] + operatorTree[9];
        return &operatorTree[10];
    }
};

#endif //PARALLELLBFGS_ROSENBROCK2D_CUH
