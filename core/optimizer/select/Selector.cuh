//
// Created by spaceman on 2022. 10. 19..
//

#ifndef PARALLELLBFGS_SELECTOR_CUH
#define PARALLELLBFGS_SELECTOR_CUH

#include "../../common/model/CudaMemoryModel.cuh"
#include "../Operator.h"

class Selector : public Operator{

public:
    Selector() {
    }
    virtual void select(CUDAConfig& cudaConfig,const double *oldX, double *newX, const double *oldF, double *newF)=0;
    virtual void printPopulationCostAtGeneration(CUDAConfig& cudaConfig, double *newF, unsigned generation)=0;
};

#endif //PARALLELLBFGS_SELECTOR_CUH