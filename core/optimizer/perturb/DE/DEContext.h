//
// Created by spaceman on 2022. 10. 15..
//

#ifndef PARALLELLBFGS_DECONTEXT_H
#define PARALLELLBFGS_DECONTEXT_H

#include "../Perturbator.h"

__global__
void differentialEvolutionStep(double *oldX, double *newX, curandState *curandState) {
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
    if (threadIdx.x == 0) {
        localA = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        while (localA == blockIdx.x) {
            localA = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        }
        localB = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        while (localB == localA || localB == blockIdx.x) {
            localB = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        }
        localC = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        while (localC == localA || localC == localB || localC == blockIdx.x) {
            localC = (int) truncf(curand_uniform(curandState + idx) * POPULATION_SIZE);
        }
        sharedR = (int) truncf(curand_uniform(curandState + idx) * X_DIM);
        sharedA = localA;
        sharedB = localB;
        sharedC = localC;
        printf("DE: a:%d,b:%d,c:%d,x:%d \n", localA, localB, localC, blockIdx.x);
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
    for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
        if (curand_uniform(curandState + idx) < CR || spanningTID == localR) {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[localA * X_DIM + spanningTID] + F * (oldX[localB * X_DIM +
                                                                                                    spanningTID] -
                                                                                               oldX[localC * X_DIM +
                                                                                                    spanningTID]);
        } else {
            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
        }
    }
//    } else {
//        for (unsigned spanningTID = threadIdx.x; spanningTID < X_DIM; spanningTID += blockDim.x) {
//            newX[blockIdx.x * X_DIM + spanningTID] = oldX[blockIdx.x * X_DIM + spanningTID];
//        }
//    }
    // newX is complete in every thread in this block
}

class DEContext : public Perturbator {
public:
    DEContext() {
        populationSize=POPULATION_SIZE;
    }
    void perturb(CUDAConfig &cudaConfig,double * dev_x1, double * dev_x2,Random* cudaRandom ) {
        differentialEvolutionStep<<<populationSize, cudaConfig.threadsPerBlock>>>(dev_x1, dev_x2, cudaRandom->dev_curandState);
    }
    double crossoverRate=CR;
    double force=F;
};

#endif //PARALLELLBFGS_DECONTEXT_H
