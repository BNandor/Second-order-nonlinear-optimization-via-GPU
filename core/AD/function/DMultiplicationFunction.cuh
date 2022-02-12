//
// Created by spaceman on 2022. 01. 22..
//

#ifndef PARALLELLBFGS_DMULTIPLICATIONFUNCTION_CUH
#define PARALLELLBFGS_DMULTIPLICATIONFUNCTION_CUH

#include "DFunction.cuh"

class DMultiplicationFunction : public DFunction {
private:
    DFunction *op1;
    DFunction *op2;
public:
    __host__ __device__
    DMultiplicationFunction(DFunction *op1, DFunction *op2) : op1(op1), op2(op2) {
        resultValue = nullptr;
    }

    __host__ __device__
    DDouble *operator()(DDouble *parameterList, unsigned size) override {
        printf("applying multiplication\n");
        if (resultValue != nullptr) {
            return resultValue;
        }
        DDouble result = *(*op1)(parameterList, size) * *(*op2)(parameterList, size);
        index = result.index;
#ifdef SAFE
        assert(index < size);
        printf("mul result index: %d\n", index);
#endif
        parameterList[index] = result;
        resultValue = &parameterList[index];
        return resultValue;
    }

};

#endif //PARALLELLBFGS_DMULTIPLICATIONFUNCTION_CUH
