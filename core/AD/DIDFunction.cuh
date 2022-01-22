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
    DFloat &operator()(DFloat *parameterList, unsigned size) override {
        assert(index < size);
        return parameterList[id];
    }
};


#endif //PARALLELLBFGS_DIDFUNCTION_CUH
