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
    virtual void perturb(CUDAConfig &cudaConfig,Model* model, Model * dev_model,double * dev_x1, double * dev_x2,double* dev_data,double* oldCosts,double* newCosts, Random* cudaRandom,
                         bool isbounded,
                         double *globalLowerBounds,
                         double *globalUpperBounds)=0;
    virtual void evaluateF(CUDAConfig &cudaConfig,Model * dev_model,double * dev_x,double* dev_data,double* costs)=0;
    int fEvaluationCount() {
        return fEvals;
    }
    void limitEvaluationsTo(int remainingEvaluations) override {
//        std::cout<<"limiting evaluations"<<std::endl;
        if(remainingEvaluations<=0){
            fEvals=0;
        }
    }
};

__global__
void snapToBounds(double *X, double *globalLowerBounds, double *globalUpperBounds, int modelsize) {
    for (unsigned spanningTID = threadIdx.x; spanningTID < modelsize; spanningTID += blockDim.x) {
        double next;
        double l;
        double u;
        next = X[blockIdx.x * modelsize + spanningTID];
        l = globalLowerBounds[spanningTID];
        if (next < l) {
            X[blockIdx.x * modelsize + spanningTID] = l;
        } else {
            u = globalUpperBounds[spanningTID];
            if (next > u) {
                X[blockIdx.x * modelsize + spanningTID] = u;
            } else {
                X[blockIdx.x * modelsize + spanningTID] = next;
            }
        }
    }
}

#endif //PARALLELLBFGS_PERTURBATOR_H
