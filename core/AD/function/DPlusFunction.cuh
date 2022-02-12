//
// Created by spaceman on 2022. 01. 22..
//

#ifndef PARALLELLBFGS_DPLUSFUNCTION_CUH
#define PARALLELLBFGS_DPLUSFUNCTION_CUH

#include "DFunction.cuh"


class DPlusFunction : public DFunction {
private:
    DFunction *op1;
    DFunction *op2;
public:
    __host__ __device__
    DPlusFunction(DFunction *op1, DFunction *op2) : op1(op1), op2(op2) {
        resultValue = nullptr;
    }

    __host__ __device__
    DDouble *operator()(DDouble *parameterList, unsigned size) override {
        if (resultValue != nullptr) {
            return resultValue;
        }
        printf("applying plus\n");
        DDouble result = *(*op1)(parameterList, size) + *(*op2)(parameterList, size);
        index = result.index;
#ifdef SAFE
        printf("plus result index: %d", index);
        assert(index < size);
#endif
        parameterList[index] = result;
        resultValue = &parameterList[index];
        return resultValue;
    }
};

#endif //PARALLELLBFGS_DPLUSFUNCTION_CUH
