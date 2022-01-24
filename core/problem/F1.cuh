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
    DDouble *costFunction(double *x, unsigned xSize, unsigned threadId) override {
        printf("initOperatorTree\n");
        initOperatorTree(x, xSize, threadId);
        printf("setting const, param references\n");
        DDouble *hundred = &operatorTree[0];
        DDouble *one = &operatorTree[1];
        DDouble *minOne = &operatorTree[2];
        DDouble *x0 = &operatorTree[3];
        DDouble *x1 = &operatorTree[4];

        printf("f1\n");
        operatorTree[5] = x0->square();
        operatorTree[6] = *x1 - operatorTree[5];
        operatorTree[7] = operatorTree[6].square();
        printf("f1\n");
        operatorTree[8] = *hundred * operatorTree[7];

        operatorTree[9] = *x0 * *minOne;
        operatorTree[10] = operatorTree[9] + *one;
        operatorTree[11] = operatorTree[10].square();
        operatorTree[12] = operatorTree[8] + operatorTree[11];
        return &operatorTree[12];
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
    void initOperatorTree(double *x, unsigned xSize, unsigned threadId) {
        operatorTree[3].value = x[2 * threadId];
        operatorTree[4].value = x[2 * threadId + 1];
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
        unsigned i = 0;
        unsigned paramsSeen = 0;
        while (i < operatorTreeSize && paramsSeen < parameterSize) {
            if (operatorTree[i].operation == ID) {
                J[paramsSeen] = operatorTree[i].derivative;
                paramsSeen++;
            }
            i++;
        }
        assert(paramsSeen == parameterSize);
    }
};

#endif //PARALLELLBFGS_F1_CUH
