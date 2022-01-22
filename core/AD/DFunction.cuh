//
// Created by spaceman on 2022. 01. 16..
//

#ifndef PARALLELLBFGS_DFUNCTION_CUHH
#define PARALLELLBFGS_DFUNCTION_CUHH

#include "DFloat.cuh"
#include <assert.h>

class DFunction {

public:
    unsigned index;
    DFloat *resultValue;

    __host__ __device__
    virtual DFloat &operator()(DFloat *parameterList, unsigned size) = 0;
};


#endif //PARALLELLBFGS_DFUNCTION_CUHH
