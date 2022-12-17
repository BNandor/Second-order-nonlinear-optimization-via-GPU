//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_PERTURBATOR_H
#define PARALLELLBFGS_PERTURBATOR_H

#include "../../../common/config/CUDAConfig.cuh"
#include "../../../common/Random.cuh"
#include "../../../common/model/BoundedParameter.cuh"
#include "../Operator.h"

class Model;
class Perturbator : public Operator {
protected:
    int fEvals=1;
public:
    int populationSize;
    virtual void perturb(CUDAConfig &cudaConfig,Model* model, Model * dev_model,double * dev_x1, double * dev_x2,double* dev_data,double* oldCosts,double* newCosts, Random* cudaRandom)=0;
    virtual void evaluateF(CUDAConfig &cudaConfig,Model * dev_model,double * dev_x,double* dev_data,double* costs)=0;
    int fEvaluationCount() {
        return fEvals;
    }
    void limitEvaluationsTo(int remainingEvaluations) override {
        if(remainingEvaluations<=0){
            fEvals=0;
        }
    }
};
#endif //PARALLELLBFGS_PERTURBATOR_H
