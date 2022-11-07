//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_DECONTEXT_H
#define PARALLELLBFGS_DECONTEXT_H

#include <curand_kernel.h>
#include "../../../common/config/CUDAConfig.cuh"
#include "../../../common/Random.cuh"
#include "../Perturbator.h"
#include "../../../common/model/BoundedParameter.cuh"
#include "../../select/Selector.cuh"
#include "../../refine/FunctionEvaluation.cuh"
// Differential Evolution Control parameters


__global__
void differentialEvolutionStep(double *oldX, double *newX, Model*model,double CR,double FORCE,curandState *curandState) {
#ifdef SAFE
    assert(blockDim.x >= 4);
#endif
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ unsigned sharedA;
    __shared__ unsigned sharedB;
    __shared__ unsigned sharedC;
    __shared__ unsigned sharedR;
    unsigned localA;
    unsigned localB;
    unsigned localC;
    unsigned localR;
    int localPopulationSize=model->populationSize;
    int localModelSize=model->modelSize;
    if (threadIdx.x == 0) {
        localA = (int) truncf(curand_uniform(curandState + idx) * (float)localPopulationSize);
        while (localA == blockIdx.x) {
            localA = (int) truncf(curand_uniform(curandState + idx) *  (float)localPopulationSize);
        }
        localB = (int) truncf(curand_uniform(curandState + idx) *  (float)localPopulationSize);
        while (localB == localA || localB == blockIdx.x) {
            localB = (int) truncf(curand_uniform(curandState + idx) *  (float)localPopulationSize);
        }
        localC = (int) truncf(curand_uniform(curandState + idx) *  (float)localPopulationSize);
        while (localC == localA || localC == localB || localC == blockIdx.x) {
            localC = (int) truncf(curand_uniform(curandState + idx) *  (float)localPopulationSize);
        }
        sharedR = (int) truncf(curand_uniform(curandState + idx) * localModelSize);
        sharedA = localA;
        sharedB = localB;
        sharedC = localC;
//        printf("DE: a:%d,b:%d,c:%d,x:%d \n", localA, localB, localC, blockIdx.x);
    }
    __syncthreads();
    localA = sharedA;
    localB = sharedB;
    localC = sharedC;
    localR = sharedR;
//  1,2,3,4
//  m[1,2,3,4]
//  1',2',3',4'
//  m[1',2',3',4']
//  min(m[1,2,3,4],m[1',2',3',4'])
//  epsilon kivétele
    // every thread has the same sharedA, sharedB, sharedC, blockId.x
//    if(blockIdx.x !=0) {
    for (unsigned spanningTID = threadIdx.x; spanningTID < localModelSize; spanningTID += blockDim.x) {
        if (curand_uniform(curandState + idx) < CR || spanningTID == localR) {
            newX[blockIdx.x * localModelSize + spanningTID] = oldX[localA * localModelSize + spanningTID] + FORCE * (oldX[localB * localModelSize +
                                                                                                           spanningTID] -
                                                                                                      oldX[localC * localModelSize +
                                                                                                    spanningTID]);
        } else {
            newX[blockIdx.x * localModelSize + spanningTID] = oldX[blockIdx.x * localModelSize + spanningTID];
        }
    }
//    } else {
//        for (unsigned spanningTID = threadIdx.x; spanningTID < localModelSize; spanningTID += blockDim.x) {
//            newX[blockIdx.x * localModelSize + spanningTID] = oldX[blockIdx.x * localModelSize + spanningTID];
//        }
//    }
    // newX is complete in every thread in this block
}

class DEContext : public Perturbator {
    void* dev_globalContext;
    void setupGlobalData(int populationSize) {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
        cudaMalloc(&dev_globalContext, sizeof(FuncEval::GlobalData)*populationSize);
        printf("Allocating %lu global memory\n",sizeof(FuncEval::GlobalData)*populationSize);
    }

public:

    DEContext() {
        populationSize=POPULATION_SIZE;
        std::unordered_map<std::string,BoundedParameter> deParams=std::unordered_map<std::string,BoundedParameter>();
        deParams["DE_CR"]=BoundedParameter(0.99,0.0,1.0);
        deParams["DE_FORCE"]=BoundedParameter(0.6,0.0,1.0);
        parameters=OperatorParameters(deParams);
        setupGlobalData(populationSize);
    }

    ~DEContext() {
        if(dev_globalContext!= nullptr) {
            cudaFree(dev_globalContext);
        }
    }

    void perturb(CUDAConfig &cudaConfig,Model *model, Model * dev_model,double * dev_x1, double * dev_x2, double* dev_data, double* oldCosts, double* newCosts, Random* cudaRandom ) override {
        differentialEvolutionStep<<<model->populationSize, cudaConfig.threadsPerBlock>>>(dev_x1, dev_x2, dev_model,
                                                                                         parameters.values["DE_CR"].value,
                                                                                         parameters.values["DE_FORCE"].value,
                                                                                         cudaRandom->dev_curandState);
        FuncEval::evaluateF<<<cudaConfig.blocksPerGrid, cudaConfig.threadsPerBlock>>>(dev_x2,dev_data,newCosts,(FuncEval::GlobalData*)dev_globalContext,dev_model);
    }
};

#endif //PARALLELLBFGS_DECONTEXT_H
