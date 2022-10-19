//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_PERTURBATOR_H
#define PARALLELLBFGS_PERTURBATOR_H

#include "DE/DEContext.h"
#include "../../common/config/CUDAConfig.cuh"
#include "../../common/Random.cuh"

class Perturbator {
public:
    int populationSize;
    virtual void perturb(CUDAConfig &cudaConfig,double * dev_x1, double * dev_x2,Random* cudaRandom)=0;
};
#endif //PARALLELLBFGS_PERTURBATOR_H
