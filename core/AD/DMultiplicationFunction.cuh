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
    DFloat *operator()(DFloat *parameterList, unsigned size) override {
        if (resultValue != nullptr) {
            return resultValue;
        }
        DFloat result = *(*op1)(parameterList, size) * *(*op2)(parameterList, size);
        index = result.index;
        assert(index < size);
        parameterList[index] = result;
        resultValue = &parameterList[index];
        return resultValue;
    }

};

#endif //PARALLELLBFGS_DMULTIPLICATIONFUNCTION_CUH
