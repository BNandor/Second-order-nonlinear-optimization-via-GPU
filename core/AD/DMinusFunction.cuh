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
    DFloat &operator()(DFloat *parameterList, unsigned size) override {
        if (resultValue != nullptr) {
            return *resultValue;
        }
        DFloat result = (*op1)(parameterList, size) - (*op2)(parameterList, size);
        index = result.index;
        assert(index < size);
        parameterList[index] = result;
        resultValue = &parameterList[index];
        return parameterList[index];
    }
};

#endif //PARALLELLBFGS_DMINUSFUNCTION_CUH
