//
// Created by spaceman on 2022. 01. 22..
//

#ifndef PARALLELLBFGS_DMINUSFUNCTION_CUH
#define PARALLELLBFGS_DMINUSFUNCTION_CUH

#include "DFunction.cuh"

class DMinusFunction : public DFunction {
private:
    DFunction *op1;
    DFunction *op2;
public:
    __host__ __device__
    DMinusFunction(DFunction *op1, DFunction *op2) : op1(op1), op2(op2) {
        resultValue = nullptr;
    }

    __host__ __device__
    DDouble *operator()(DDouble *parameterList, unsigned size) override {
        printf("applying minus\n");
        if (resultValue != nullptr) {
            return resultValue;
        }
        DDouble result = *(*op1)(parameterList, size) - *(*op2)(parameterList, size);
        index = result.index;
#ifdef SAFE
        printf("minus result index: %d", index);
        assert(index < size);
#endif
        parameterList[index] = result;
        resultValue = &parameterList[index];
        return resultValue;
    }
};

#endif //PARALLELLBFGS_DMINUSFUNCTION_CUH
