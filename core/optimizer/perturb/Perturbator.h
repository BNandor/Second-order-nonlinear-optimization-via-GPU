//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_PERTURBATOR_H
#define PARALLELLBFGS_PERTURBATOR_H

#include "../../common/config/CUDAConfig.cuh"
#include "../../common/Random.cuh"
#include "../../common/model/BoundedParameter.cuh"

class Model;
class Perturbator {

public:
    int populationSize;
    OperatorParameters parameters;
    virtual void perturb(CUDAConfig &cudaConfig,Model* model, Model * dev_model,double * dev_x1, double * dev_x2,double* oldCosts, Random* cudaRandom)=0;
};
#endif //PARALLELLBFGS_PERTURBATOR_H
