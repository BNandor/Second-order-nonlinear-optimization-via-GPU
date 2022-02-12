//
// Created by spaceman on 2022. 01. 23..
//

#ifndef PARALLELLBFGS_DSQUAREFUNCTION_CUH
#define PARALLELLBFGS_DSQUAREFUNCTION_CUH

#include "DFunction.cuh"

class DSquareFunction : public DFunction {
private:
    DFunction *op1;
public:
    __host__ __device__
    DSquareFunction(DFunction *op1) : op1(op1) {
        resultValue = nullptr;
    }

    __host__ __device__
    DDouble *operator()(DDouble *parameterList, unsigned size) override {
        printf("applying square\n");
        if (resultValue != nullptr) {
            return resultValue;
        }
        DDouble result = (*op1)(parameterList, size)->square();
        index = result.index;

#ifdef SAFE
        printf("square result index: %d", index);
        assert(index < size);
#endif
        parameterList[index] = result;
        resultValue = &parameterList[index];
        return resultValue;
    }

};

#endif //PARALLELLBFGS_DSQUAREFUNCTION_CUH
