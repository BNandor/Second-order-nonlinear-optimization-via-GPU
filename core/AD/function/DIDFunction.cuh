//
// Created by spaceman on 2022. 01. 22..
//

#ifndef PARALLELLBFGS_DIDFUNCTION_CUH
#define PARALLELLBFGS_DIDFUNCTION_CUH

#include "DFunction.cuh"

class DIDFunction : public DFunction {
private:
    unsigned id;
public:
    __host__ __device__
    DIDFunction(unsigned id) : id(id) {
        index = id;
        resultValue = nullptr;
    }

    __host__ __device__
    DDouble *operator()(DDouble *parameterList, unsigned size) override {
#ifdef SAFE
        printf("applying id\n");
        assert(index < size);
        printf("id result index: %d", index);
#endif
        return &parameterList[id];
    }
};


#endif //PARALLELLBFGS_DIDFUNCTION_CUH
