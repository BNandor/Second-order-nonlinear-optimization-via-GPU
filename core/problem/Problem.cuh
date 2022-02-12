//
// Created by spaceman on 2022. 01. 23..
//

#ifndef PARALLELLBFGS_PROBLEM_CUH
#define PARALLELLBFGS_PROBLEM_CUH

#include "../AD/DDouble.cuh"

class Problem {
public:
    unsigned operatorTreeSize;
    unsigned parameterSize;
    unsigned constantSize;
    unsigned globalIndex = 0;
    double *J;
    DDouble *operatorTree;


    __device__ __host__
    void clearDerivatives() {
        for (unsigned i = 0; i < operatorTreeSize; i++) {
            operatorTree[i].derivative = 0.0;
        }
    }

    __device__ __host__ void evalJacobian() {
        clearDerivatives();
        operatorTree[operatorTreeSize - 1].setPartialDerivatives(operatorTree);
        for (unsigned i = constantSize; i < constantSize + parameterSize; i++) {
            J[i - constantSize] = operatorTree[i].derivative;
        }
    }

    __device__ __host__
    void evalStep(double *x, double *xNext, unsigned xSize, double *jacobian, double alpha) {
#ifdef SAFE
        assert(xSize == parameterSize);
#endif
        for (unsigned i = 0; i < xSize; i++) {
            xNext[i] = x[i] - alpha * jacobian[i];
        }
    }

    __device__ __host__
    void initOperatorTree(double *x, unsigned xSize) {
#ifdef SAFE
        assert(xSize == parameterSize);
#endif
        for (unsigned i = 0; i < parameterSize; i++) {
            operatorTree[constantSize + i].value = x[i];
        }
        globalIndex = parameterSize + constantSize;
    }

    __device__ __host__
    void initIndex() {
        for (unsigned i = 0; i < constantSize + parameterSize; i++) {
            operatorTree[i].index = i;
            operatorTree[i].globalIndex = &globalIndex;
        }
    }

    __device__ __host__
    void initConst(double *constants, unsigned constantsSize) {
#ifdef SAFE
        assert(constantsSize == this->constantSize);
#endif
        for (unsigned i = 0; i < constantSize; i++) {
            operatorTree[i].operation = CONST;
            operatorTree[i].value = constants[i];
        }
    }
};

#endif //PARALLELLBFGS_PROBLEM_CUH
