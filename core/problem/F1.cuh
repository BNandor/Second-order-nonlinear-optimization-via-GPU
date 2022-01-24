//
// Created by spaceman on 2022. 01. 23..
//

#ifndef PARALLELLBFGS_F1_CUH
#define PARALLELLBFGS_F1_CUH

#include "Problem.cuh"
#include "../AD/function/DFunction.cuh"
#include "../AD/function/DSquareFunction.cuh"
#include "../AD/DDouble.cuh"

class F1 : public Problem {
public:
    static const unsigned operatorTreeSize = 13;
    static const unsigned parameterSize = 2;
    static const unsigned constantSize = 3;
    unsigned globalIndex = 0;
    double J[parameterSize];
    DDouble operatorTree[operatorTreeSize] = {};

    __device__ __host__
    F1() {
        initConst();
    }

    __device__ __host__
    DDouble *eval(double *x, unsigned xSize) {
        initOperatorTree(x, xSize);
        DDouble *hundred = &operatorTree[0];
        DDouble *one = &operatorTree[1];
        DDouble *minOne = &operatorTree[2];
        DDouble *x0 = &operatorTree[3];
        DDouble *x1 = &operatorTree[4];

        operatorTree[5] = x0->square();
        operatorTree[6] = *x1 - operatorTree[5];
        operatorTree[7] = operatorTree[6].square();
        operatorTree[8] = *hundred * operatorTree[7];

        operatorTree[9] = *x0 * *minOne;
        operatorTree[10] = operatorTree[9] + *one;
        operatorTree[11] = operatorTree[10].square();
        operatorTree[12] = operatorTree[8] + operatorTree[11];
        return &operatorTree[12];
    }

    __device__ __host__
    void evalStep(double *x, double *xNext, unsigned xSize, double alpha) {
        assert(xSize == parameterSize);
        for (unsigned i = 0; i < xSize; i++) {
            xNext[i] = x[i] - alpha * J[i];
        }
    }

    __device__ __host__
    void initConst() {
        operatorTree[0].value = 100.0;
        operatorTree[1].value = 1.0;
        operatorTree[2].value = -1.0;
        for (unsigned i = 0; i < constantSize; i++) {
            operatorTree[i].operation = CONST;
        }
        for (unsigned i = 0; i < constantSize + parameterSize; i++) {
            operatorTree[i].index = i;
            operatorTree[i].globalIndex = &globalIndex;
        }
    }

    __device__ __host__
    void initOperatorTree(double *x, unsigned xSize) {
        assert(xSize == parameterSize);
        for (unsigned i = 0; i < parameterSize; i++) {
            operatorTree[constantSize + i].value = x[i];
        }
        globalIndex = parameterSize + constantSize;
    }

    __device__ __host__
    void clearDerivatives() {
        for (unsigned i = 0; i < operatorTreeSize; i++) {
            operatorTree[i].derivative = 0.0;
        }
    }

    __device__ __host__ void setJacobian() {
        clearDerivatives();
        operatorTree[operatorTreeSize - 1].setPartialDerivatives(operatorTree);
        for (unsigned i = constantSize; i < constantSize + parameterSize; i++) {
            J[i - constantSize] = operatorTree[i].derivative;
        }
    }
};

#endif //PARALLELLBFGS_F1_CUH
