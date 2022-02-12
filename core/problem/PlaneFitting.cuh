//
// Created by spaceman on 2022. 01. 24..
//

#ifndef PARALLELLBFGS_PLANEFITTING_CUH
#define PARALLELLBFGS_PLANEFITTING_CUH

#include "Problem.cuh"
#include "../AD/function/DFunction.cuh"
#include "../AD/function/DSquareFunction.cuh"
#include "../AD/DDouble.cuh"

class PlaneFitting : public Problem {
public:
    static const unsigned ThisOperatorTreeSize = 12;
    static const unsigned ThisParameterSize = 3;
    static const unsigned ThisConstantSize = 3;
    double ThisJ[ThisParameterSize];
    DDouble ThisOperatorTree[ThisOperatorTreeSize] = {};

    __device__ __host__
    PlaneFitting() {
        operatorTreeSize = ThisOperatorTreeSize;
        parameterSize = ThisParameterSize;
        constantSize = ThisConstantSize;
        operatorTree = ThisOperatorTree;
        J = ThisJ;
    }

    __device__ __host__
    void setConstants(double *constants, unsigned constantsSize) {
        initConst(constants, constantsSize);
        initIndex();
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
//        Math.pow((point.pz - (x(0)*point.x + x(1) * point.py + x(2))),2)
        initOperatorTree(x, xSize);
        DDouble *cx = &operatorTree[0];
        DDouble *cy = &operatorTree[1];
        DDouble *cz = &operatorTree[2];
        DDouble *px = &operatorTree[3];
        DDouble *py = &operatorTree[4];
        DDouble *pz = &operatorTree[5];

        operatorTree[6] = *px * *cx;
        operatorTree[7] = *py * *cy;
        operatorTree[8] = operatorTree[6] + operatorTree[7];
        operatorTree[9] = operatorTree[8] + *pz;
        operatorTree[10] = *cz - operatorTree[9];
        operatorTree[11] = operatorTree[10].square();
        return &operatorTree[11];
    }
};


#endif //PARALLELLBFGS_PLANEFITTING_CUH
